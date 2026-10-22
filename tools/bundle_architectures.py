"""Extend the existing web bundle without changing the framework-parity models."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'web/bundle';catalog=json.loads((out/'catalog.json').read_text(encoding='utf-8'))
manifest={}
for key in catalog:
    if key in ['diabetes','housing']:
        folder=ROOT/'results'/key/'classical'
        names=['LogisticRegression' if key=='diabetes' else 'Ridge','RandomForest']
    else:
        folder=ROOT/'results/architectures'/key
        names=['basic_cnn','vgg_style','mobilenet_style','resnet_style'] if key in ['mnist','eurosat'] else ['simple_rnn','lstm','gru','bilstm']
    manifest[key]={}
    for name in names:
        config=json.loads((folder/f'{name}_config.json').read_text(encoding='utf-8'))
        result=json.loads((folder/f'{name}.json').read_text(encoding='utf-8'))
        file=f'{key}_model_{name}.npz';shutil.copyfile(folder/f'{name}.npz',out/file)
        manifest[key][name]=dict(config=config,file=file,label=config.get('label',name),framework='scikit-learn' if key in ['diabetes','housing'] else 'pytorch',validation=result['validation'],test=result['test'])
(out/'architectures.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Bundled',sum(len(v) for v in manifest.values()),'additional architecture/classical checkpoints')
