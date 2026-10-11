from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PLIFMix.ui.task_runner import LatestTaskRunner, TaskRunner, close_task_runners


class FakeOwner:
    def __init__(self):
        self.jobs = {}
        self.next_id = 0
        self.state_model = SimpleNamespace(save=Mock())

    def winfo_toplevel(self):
        return self

    def bind(self, *args, **kwargs):
        pass

    def after(self, milliseconds, callback):
        self.next_id += 1
        self.jobs[self.next_id] = callback
        return self.next_id

    def after_cancel(self, job):
        self.jobs.pop(job, None)

    def advance(self):
        job = next(iter(self.jobs))
        self.jobs.pop(job)()


class TaskRunnerTests(unittest.TestCase):
    def setUp(self):
        self.owner = FakeOwner()
        self.log = Mock()

    def finish(self, runner):
        runner._thread.join(timeout=2)
        self.assertFalse(runner.worker_running)
        self.owner.advance()

    def test_batch_rejects_duplicate_submission(self):
        runner = TaskRunner(self.owner, self.log, latest_only=False)
        work = Mock(return_value=12)
        callback = Mock()
        self.assertTrue(runner.submit(work, callback, "start"))
        self.assertFalse(runner.submit(work, callback, "start again"))
        self.owner.advance()
        self.finish(runner)
        work.assert_called_once()
        callback.assert_called_once_with(12)
        self.assertFalse(runner.busy)

    def test_latest_pending_request_replaces_intermediate_request(self):
        runner = LatestTaskRunner(self.owner, self.log)
        callbacks = []
        runner.submit(lambda: 1, callbacks.append, "first")
        self.owner.advance()
        runner.submit(lambda: 2, callbacks.append, "second")
        runner.submit(lambda: 3, callbacks.append, "third")
        self.finish(runner)
        self.finish(runner)
        self.assertEqual(callbacks, [1, 3])

    def test_failed_display_callback_does_not_stall_pending_work(self):
        runner = LatestTaskRunner(self.owner, self.log)
        runner.submit(lambda: 1, Mock(side_effect=ValueError("display error")), "first")
        self.owner.advance()
        callback = Mock()
        runner.submit(lambda: 2, callback, "second")
        self.finish(runner)
        self.finish(runner)
        callback.assert_called_once_with(2)
        self.assertFalse(runner.busy)

    def test_close_cancels_pending_callbacks_but_tracks_active_worker(self):
        runner = LatestTaskRunner(self.owner, self.log)
        release = Event()
        self.addCleanup(release.set)
        callback = Mock()
        runner.submit(lambda: release.wait(timeout=3), callback, "work")
        self.owner.advance()
        runner.submit(lambda: 2, callback, "pending")
        selected = close_task_runners(self.owner)
        self.assertIn(runner, selected)
        self.assertTrue(runner.worker_running)
        self.assertFalse(self.owner.jobs)
        release.set()
        runner._thread.join(timeout=2)
        callback.assert_not_called()
        self.assertFalse(runner.submit(lambda: 3, callback, "closed"))


if __name__ == "__main__":
    unittest.main()
