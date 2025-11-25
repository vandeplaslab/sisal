def test_import():
    from sisal import __version__

    assert isinstance(__version__, str)
