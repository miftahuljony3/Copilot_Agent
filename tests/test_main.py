from main import _config_path_from_argv


def test_config_path_from_separate_option():
    assert str(_config_path_from_argv(["--config", "/etc/agent.yaml", "chat"])) == (
        "/etc/agent.yaml"
    )


def test_config_path_from_equals_option():
    assert str(_config_path_from_argv(["--config=custom.yaml", "chat"])) == (
        "custom.yaml"
    )


def test_config_path_defaults_to_repository_config():
    assert str(_config_path_from_argv(["chat"])) == "config.yaml"
