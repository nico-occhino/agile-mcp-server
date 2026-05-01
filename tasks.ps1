param (
    [Parameter(Position=0)]
    [ValidateSet("install", "test", "eval", "schema", "run")]
    [string]$Task = "run"
)

switch ($Task) {
    "install" {
        pip install -e ".[dev,eval]"
    }
    "test" {
        pytest tests/
    }
    "eval" {
        python -m scripts.eval
    }
    "schema" {
        python -m scripts.export_schema
    }
    "run" {
        python -m client.main
    }
}
