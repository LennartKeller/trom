"""Tests for the trom CLI."""

from click.testing import CliRunner

from trom.cli import main


class TestCli:
    def test_romanize_argument(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-l", "rus", "Привет мир"])
        assert result.exit_code == 0
        assert result.output.strip() == "Privet mir"

    def test_romanize_stdin(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-l", "rus"], input="Привет\nмир\n")
        assert result.exit_code == 0
        assert result.output.splitlines() == ["Privet", "mir"]

    def test_uroman_romanizer(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-l", "ell", "-r", "uroman", "Γεια"])
        assert result.exit_code == 0
        assert result.output.strip() != "Γεια"

    def test_invalid_lang(self):
        runner = CliRunner()
        result = runner.invoke(main, ["-l", "xyz", "hello"])
        assert result.exit_code != 0
        assert "not found in ISO 639" in result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Romanize TEXT" in result.output
