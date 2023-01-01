from typing import Optional
import unittest

import nbclient
from nbclient.exceptions import CellExecutionError
import nbformat


class TestNotebooks(unittest.TestCase):
    @staticmethod
    def _execute_notebook_from_path(path_to_notebook: str) -> None:
        """Execute notebook from path."""
        notebook = nbformat.read(path_to_notebook, nbformat.NO_CONVERT)
        nbclient.execute(notebook)

    def _check_notebook_raises(
        self,
        path_to_notebook: str,
        expected_error_name: str,
        expected_error_value: Optional[str] = None,
    ) -> None:
        """Check notebook raises a particular error."""
        try:
            self._execute_notebook_from_path(path_to_notebook)
        except CellExecutionError as err:
            self.assertEqual(expected_error_name, err.ename)
            if expected_error_value:
                self.assertEqual(expected_error_value, err.evalue)

    def test_notebooks_execute(self):
        notebooks_that_should_execute = [
            "docs/modules/chains/getting_started.ipynb",
            "docs/modules/chains/examples/llm_bash.ipynb",
        ]
        for path in notebooks_that_should_execute:
            self._execute_notebook_from_path(path)

        self._check_notebook_raises(
            "docs/modules/chains/examples/moderation.ipynb",
            "ValueError",
            "Text was found that violates OpenAI's content policy.",
        )
