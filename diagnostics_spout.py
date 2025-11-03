import importlib, inspect, pkgutil, sys, os
try:
    import SpoutGL as spout
except Exception as e:
    print("IMPORT ERROR:", e)
    sys.exit(1)

print("spout module file:", getattr(spout, "__file__", "<built-in>"))
print("spout __package__:", getattr(spout, "__package__", None))
print("spout __all__:", getattr(spout, "__all__", None))

print("\n--> dir(spout):")
print(sorted([n for n in dir(spout) if not n.startswith('_')]))

print("\n--> pkgutil.iter_modules in spout folder (if any):")
spout_pkg_path = os.path.dirname(getattr(spout, "__file__", ""))
for finder, name, ispkg in pkgutil.iter_modules([spout_pkg_path]):
    print("  -", name, "ispkg=", ispkg)

print("\n--> show top of spout/__init__.py (first 200 lines):")
try:
    with open(getattr(spout, "__file__"), "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i>=200: break
            print(line.rstrip())
except Exception as e:
    print("Could not read file:", e)

print("\n--> Attempt to inspect callables in module:")
for name in sorted(dir(spout)):
    if name.startswith('_'): continue
    obj = getattr(spout, name)
    if inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismodule(obj):
        try:
            sig = inspect.signature(obj)
            print(f"{name}  signature: {sig}")
        except Exception:
            print(f"{name}  (callable, signature unknown)")