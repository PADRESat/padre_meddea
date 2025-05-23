# Force MPL to use non-gui backends for testing.
try:
    import matplotlib
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
    matplotlib.use("Agg")
except ImportError:
    HAVE_MATPLOTLIB = False
