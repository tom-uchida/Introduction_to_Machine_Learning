## 2-3_Regularization

### 概要
- 正則化とは線形回帰モデルのパラメータ$w$が訓練データに過剰にフィットし，過学習する現象を防ぐために使用する．
- 過学習時には，訓練データに過剰にフィットさせるためにパラメータ$w$の一部の値(絶対値)が大きくなりすぎるといったことが発生する．
   - ここでパラメータ$w$の値(絶対値)が大きくなりすぎないようにペナルティを与えるテクニックが正則化．
- 正則化では回帰モデルの汎化性能(テストデータに対する$R^2$スコア)を向上させることが目標．

### L1正則化とL2正則化の使い分け
- L1正則化とL2正則化を使い分けるポイントは，重みパラメータの一部を0にし，回帰モデルで使用される特徴量の数を制限したいかどうか．
   - L2正則化：過学習を防ぎたいが，使用する特徴量のすべてが重要な場合
   - L1正則化：過学習を防ぐために，使用する特徴量が減っても良い場合