# Fake pandas module to bypass strict Windows DLL restrictions
class DataFrame: pass
class Series: pass

__version__ = "9.9.9"
__spec__ = None
