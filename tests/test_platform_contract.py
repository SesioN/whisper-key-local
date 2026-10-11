import ast
from conftest import PLATFORM_DIR, defined_names, platform_module_names

SOURCE_DIR = PLATFORM_DIR.parent


def platform_attributes_used_by_shared_code() -> dict:
    modules = set(platform_module_names())
    used = {}
    for source_path in SOURCE_DIR.rglob("*.py"):
        if PLATFORM_DIR in source_path.parents:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("platform"):
                aliases.update({alias.asname or alias.name: alias.name for alias in node.names if alias.name in modules})
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                used.setdefault(aliases[node.value.id], set()).add(node.attr)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-2:-1] == ["platform"]:
                module_name = node.module.split(".")[-1]
                if module_name in modules:
                    used.setdefault(module_name, set()).update(alias.name for alias in node.names)
    return used


def exported_platform_modules() -> set:
    tree = ast.parse((PLATFORM_DIR / "__init__.py").read_text(encoding="utf-8"))
    return {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module in ("macos", "windows")
            for alias in node.names}


def test_every_exported_platform_module_exists_on_both_platforms():
    exported = exported_platform_modules()
    assert exported
    for platform_name in ("windows", "macos"):
        assert exported <= {path.stem for path in (PLATFORM_DIR / platform_name).glob("*.py")}


def test_shared_code_only_uses_platform_functions_both_platforms_define():
    used = platform_attributes_used_by_shared_code()
    assert used
    missing = {}
    for module_name, attributes in used.items():
        for platform_name in ("windows", "macos"):
            absent = attributes - defined_names(PLATFORM_DIR / platform_name / f"{module_name}.py")
            if absent:
                missing[f"{platform_name}/{module_name}"] = sorted(absent)
    assert missing == {}


def test_platform_package_exports_every_windows_module():
    assert set(platform_module_names()) <= exported_platform_modules()
