"""Unit tests for the ROCm `CustomAllreduce` under expandable segments."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_real_empty = torch.empty
_CustomAllreduce = custom_all_reduce.CustomAllreduce


class _FakeHip:
    """A caching allocator and IPC layer that, like HIP, cannot export VMM memory."""

    def __init__(self, expandable: bool):
        self.expandable = expandable
        self.vmm_tensors = []
        self.graph_inputs = []

    def snapshot(self):
        return {"allocator_settings": {"expandable_segments": self.expandable}}

    def set_allocator_settings(self, conf: str):
        key, value = conf.split(":")
        assert key == "expandable_segments", conf
        self.expandable = value == "True"

    def empty(self, *args, **kwargs):
        tensor = _real_empty(*args, **kwargs)
        if self.expandable:
            self.vmm_tensors.append(tensor)
        return tensor

    def get_ipc_handle(self, tensor):
        if any(tensor is vmm for vmm in self.vmm_tensors):
            raise RuntimeError("hipIpcGetMemHandle: invalid argument")
        return b"handle"

    def get_graph_buffer_ipc_meta(self, ptr):
        handles = b"".join(self.get_ipc_handle(t) for t in self.graph_inputs)
        return handles, [0] * len(self.graph_inputs)

    def ops(self):
        return SimpleNamespace(
            IS_CUSTOM_AR_AVAILABLE=True,
            meta_size=lambda: 0,
            allocate_meta_buffer=lambda size: _real_empty(size, dtype=torch.uint8),
            get_meta_buffer_ipc_handle=self.get_ipc_handle,
            init_custom_ar=lambda *args: 1,
            register_buffer=lambda ptr, inp, handles, offsets: None,
            all_reduce_reg=lambda ptr, inp, out: self.graph_inputs.append(inp),
            all_reduce_unreg=lambda ptr, inp, buffer, out: None,
            get_graph_buffer_ipc_meta=self.get_graph_buffer_ipc_meta,
            register_graph_buffers=lambda ptr, handles, offsets: None,
        )


class TestCustomAllreduceExpandableSegments(CustomTestCase):
    def _build(self, *, expandable: bool):
        hip = _FakeHip(expandable)
        patches = [
            patch.object(custom_all_reduce, "_is_hip", True),
            patch.object(custom_all_reduce, "ops", hip.ops()),
            patch.object(
                custom_all_reduce, "_use_amd_deterministic_impl", lambda: False
            ),
            patch.object(
                custom_all_reduce,
                "can_use_custom_all_reduce_with_nvlink",
                lambda **kwargs: True,
            ),
            patch.object(custom_all_reduce.dist, "get_rank", lambda group: 0),
            patch.object(custom_all_reduce.dist, "get_world_size", lambda group: 2),
            patch.object(
                _CustomAllreduce,
                "_gather_ipc_meta",
                lambda self, shard_data: ([shard_data[0]] * 2, [shard_data[1]] * 2),
            ),
            patch.object(torch.cuda.memory, "_snapshot", hip.snapshot),
            patch.object(
                torch.cuda.memory, "_set_allocator_settings", hip.set_allocator_settings
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", lambda: True),
            patch.object(torch, "empty", hip.empty),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        ca = _CustomAllreduce(group=object(), device="cpu", max_size=1024)
        # `close()` would otherwise reach the real ops from `__del__`.
        self.addCleanup(setattr, ca, "_ptr", 0)
        return ca, hip

    def _capture_all_reduce(self, ca, hip):
        # A tensor captured into a graph comes from the caching allocator.
        inp = hip.empty(64, dtype=torch.bfloat16)
        with ca.capture():
            self.assertIsNotNone(ca.custom_all_reduce(inp))

    def test_init_and_capture_avoid_exporting_vmm_memory(self):
        ca, hip = self._build(expandable=True)
        self.assertFalse(ca.disabled)
        self.assertTrue(hip.expandable, "the allocator setting was not restored")

        self._capture_all_reduce(ca, hip)
        self.assertEqual(hip.graph_inputs, [])

    def test_regular_segments_keep_registering_graph_inputs(self):
        ca, hip = self._build(expandable=False)
        self.assertFalse(hip.expandable)

        self._capture_all_reduce(ca, hip)
        self.assertEqual(len(hip.graph_inputs), 1)


if __name__ == "__main__":
    unittest.main()
