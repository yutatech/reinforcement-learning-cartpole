# reinforcement-learning-cartpole
Deep Q-NetworkによるCart Pole制御を実装しました。物理シミュレーション環境はgymnasiumです。

## 実行環境
MacBook Air (M1)

## 実行方法
```shell
# setup
python3 -m venv cartpole
source cartpole/bin/activate

pip install gymnasium torch pygame moviepy

# run
mkdir video # レンダリング映像出力ディレクトリ
python3 ./cartpole.py
```

venv仮想環境を非有効化
```shell
deactivate
```

## 参考文献
- [Deep Q Network(DQN)をPyTorchで実装](https://qiita.com/Rowing0914/items/eeba790401bcaf2c723c)
- [PyTorchで深層強化学習（DQN、DoubleDQN）を実装してみた](https://ie110704.net/2017/10/15/pytorchで深層強化学習（dqn、doubledqn）を実装してみた/)