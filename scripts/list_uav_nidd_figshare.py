import json, urllib.request, os, struct, zipfile, io
D=os.path.expanduser("~/datasets/uav_nidd/figshare"); os.makedirs(D, exist_ok=True)
art=json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/25486462"))
files={f["name"]:f for f in art["files"]}
for f in files.values(): print(f["name"], f["size"], f.get("computed_md5"), f["download_url"])

# 1) small file: full download
small=files["GCS-to-UAV Updated.zip"]; p=os.path.join(D,small["name"])
#(os.path.exists(p) and os.path.getsize(p)==small["size"]) or urllib.request.urlretrieve(small["download_url"], p)
#print("downloaded", p, os.path.getsize(p))

# 2) big file: list via HTTP range requests (central directory only)
class RangeFile(io.RawIOBase):
    def __init__(s,url,size): s.url,s.size,s.pos,s.bytes=url,size,0,0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s,o,w=0):
        s.pos = o if w==0 else (s.pos+o if w==1 else s.size+o); return s.pos
    def readinto(s,b):
        n=min(len(b), s.size-s.pos)
        if n<=0: return 0
        req=urllib.request.Request(s.url, headers={"Range":f"bytes={s.pos}-{s.pos+n-1}"})
        data=urllib.request.urlopen(req).read(); b[:len(data)]=data; s.pos+=len(data); s.bytes+=len(data); return len(data)
big=files["UAV-NIDD-UAG.zip"]
url=big["download_url"]
rf=RangeFile(url, big["size"])
z=zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1<<16))
print(f"\n{big['name']}: {len(z.infolist())} entries (read {rf.bytes/1e3:.0f} KB to list)")
#for i in z.infolist(): print(f"{i.file_size:>14,}  {i.compress_size:>14,}  {i.date_time}  {i.filename}")

print("\n== same CRC32 + size in more than one place (captures reused across scenarios)")
from collections import defaultdict
g=defaultdict(list)
for i in z.infolist():
    if i.file_size>0 and i.filename.lower().endswith((".pcap",".cap",".pcapng.gz",".csv")): g[(i.CRC,i.file_size)].append(i.filename.replace("UAV-NIDD-UAG/UAV-NIDD/",""))
for (crc,sz),names in sorted(g.items(), key=lambda kv:-kv[0][1]):
    if len(names)>1: print(f"{sz:>14,}  crc={crc:08x}"); [print("     ",n) for n in names]
