from contextlib import contextmanager

@contextmanager
def elapse_time():
	import time
	start = time.time()
	yield
	print(f"Elapsed time: {time.time() - start}")