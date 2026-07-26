import asyncio, threading, time, os, sys
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
res = {}

def work():
    t = time.time()
    try:
        c = chromadb.PersistentClient(path="./data/chroma_test")
        col = c.get_or_create_collection("x", embedding_function=None)
        res["ok"] = (col.count(), round(time.time() - t, 2))
    except Exception as e:
        res["err"] = repr(e)

async def main():
    th = threading.Thread(target=work)
    th.start()
    for i in range(25):
        await asyncio.sleep(1)
        if not th.is_alive():
            break
        print("loop alive, waiting", i); sys.stdout.flush()
    print("RESULT", res); sys.stdout.flush()

asyncio.run(main())
