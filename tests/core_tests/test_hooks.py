"""Tests for Post-Analysis Hook System"""

import sys
from pathlib import Path
from typing import cast

sys.path.append(str(Path(__file__).parent.parent))


class TestAfterAnalysisHooks:
    """Test cases for the post-analysis hook system"""

    def test_run_hooks_with_no_enabled_hooks(self, tmp_path):
        """Test hook system when all hooks are disabled"""
        from unittest.mock import patch

        from after_analysis import run_hooks
        from newsbot.models import AnalysisData

        # Create a temporary report file
        report_path = tmp_path / "test_report.html"
        report_path.write_text("<html><body>Test</body></html>")

        analysis_data: AnalysisData = cast(
            AnalysisData,
            {
                "success": True,
                "articles_count": 100,
                "stories_count": 10,
            },
        )

        # Mock email hook execution to prevent actual emails
        with patch("after_analysis.email_sender.execute", return_value=None):
            # Should not raise any errors when all hooks are disabled (start with _)
            run_hooks(report_path, analysis_data)

    def test_run_hooks_with_none_analysis_data(self, tmp_path):
        """Test that None analysis_data is handled gracefully"""
        from unittest.mock import patch

        from after_analysis import run_hooks

        report_path = tmp_path / "test_report.html"
        report_path.write_text("<html><body>Test</body></html>")

        # Mock email hook execution to prevent actual emails
        with patch("after_analysis.email_sender.execute", return_value=None):
            # Should not raise error with None
            run_hooks(report_path, None)

    def test_run_hooks_with_empty_analysis_data(self, tmp_path):
        """Test with empty analysis data dict"""
        from unittest.mock import patch

        from after_analysis import run_hooks

        report_path = tmp_path / "test_report.html"
        report_path.write_text("<html><body>Test</body></html>")

        # Mock email hook execution to prevent actual emails
        with patch("after_analysis.email_sender.execute", return_value=None):
            # Should not raise error with empty dict
            run_hooks(report_path, {})


class TestHookSystemIntegration:
    """Integration tests for the hook system"""

    def test_example_hook_files_exist(self):
        """Test that example hook files exist in the after_analysis directory"""
        from pathlib import Path

        hooks_dir = (
            Path(__file__).parent.parent.parent / "src" / "after_analysis"
        )

        # Check that the after_analysis directory exists
        assert hooks_dir.exists(), "after_analysis directory should exist"

        # Check that __init__.py exists (the hook system)
        init_file = hooks_dir / "__init__.py"
        assert init_file.exists(), "Hook system __init__.py should exist"

        # Check that at least one hook file exists (email_sender.py or _email_sender.py)
        email_enabled = (hooks_dir / "email_sender.py").exists()
        email_disabled = (hooks_dir / "_email_sender.py").exists()
        assert email_enabled or email_disabled, (
            "Email sender hook should exist (enabled or disabled)"
        )

    def test_hook_system_module_exports(self):
        """Test that hook system exports run_hooks function"""
        from after_analysis import run_hooks

        assert run_hooks is not None, "Should export run_hooks"
        assert callable(run_hooks), "run_hooks should be callable"


class TestReportAgentReturnsPath:
    """Test that report agent returns file path"""

    def test_save_report_returns_path(self, tmp_path):
        """Test that _save_report returns the file path"""
        from unittest.mock import patch

        from newsbot.agents.report_agent import ReportGeneratorAgent
        from utilities.models import ConfigModel, ReportConfigModel

        config = ConfigModel(
            name="Test",
            report=ReportConfigModel(format="html"),
        )

        agent = ReportGeneratorAgent(config)

        # Patch os operations
        with patch("os.makedirs"):
            with patch("builtins.open", create=True) as mock_open:
                mock_file = mock_open.return_value.__enter__.return_value

                report_path = agent._save_report(
                    "<html>Test</html>", folder="reports/Test"
                )

                assert isinstance(report_path, Path)
                assert "news_report_" in str(report_path)
                assert str(report_path).endswith(".html")


class TestPipelineIntegration:
    """Test that pipeline integrates with hook system"""

    def test_results_has_report_path_field(self):
        """Test that Results class has report_path field"""
        from newsbot.pipeline import Results

        results = Results()
        assert hasattr(results, "report_path"), (
            "Results should have report_path field"
        )
        assert results.report_path == "", (
            "Default report_path should be empty string"
        )
