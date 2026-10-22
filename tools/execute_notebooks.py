from pathlib import Path
import os,json,sys
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
kernel=ROOT/'tmp/jupyter/kernels/tieuluan';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (Tiểu luận)','language':'python'}))
os.environ['JUPYTER_PATH']=str(ROOT/'tmp/jupyter');os.environ['IPYTHONDIR']=str(ROOT/'tmp/ipython')
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    if len(sys.argv)>1 and not any(key in path.name for key in sys.argv[1:]):continue
    book=nbformat.read(path,as_version=4)
    def completed(cell,cell_index,execute_reply):
        nbformat.write(book,path);print(path.name,'cell',cell_index+1,'done',flush=True)
    client=NotebookClient(book,timeout=7200,kernel_name='tieuluan',resources={'metadata':{'path':str(ROOT)}},on_cell_executed=completed)
    try:client.execute()
    finally:nbformat.write(book,path)
    print('FINISHED',path.name,flush=True)
