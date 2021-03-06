## 第1章 機械学習の概要

### 1.3 機械学習の各種法のアルゴリズムを学ぶ意義
1. 機械学習システムを構築するうえで，理想的な性能の機械学習モデルの実現に向けて自ら試行錯誤できるようになるため．
   - 機械学習ライブラリを使用して性能が出なかった際に，次に何をすべきなのか，どのようなアプローチをとれば性能を改善できるかの試行錯誤をできるようにするため．
2. 各アルゴリズムを使用する際のハイパーパラメータの調整を適切に行うため．
   - 各手法の各ハイパーパラメータの設定値によって，機械学習の性能は変化する．
   - 各アルゴリズムにおけるハイパーパラメータの役割と値を変化させたときの挙動のイメージをつかんでおくことが大切．
   - 実装した機械学習モデルの性能が出ないときに闇雲にハイパーパラメータを調整するのではなく，アルゴリズムとハイパーパラメータの役割を理解したうえで設定を変更できるようになるため．
3. 適切なアルゴリズムを選択し，必要に応じて，機械学習モデルの内容や結果を説明できるようになるため．
   - 機械学習をビジネスに適用する際には，使用したアルゴリズムや結果に対する説明を求められる場合が多い．
   - 使用した機械学習の手法のアルゴリズムの性質に基づいて，結果に対する示唆や根拠を示すことが求められる．

### 1.4 機械学習の勉強方法
#### 機械学習の勉強を効率的に進めるコツ
1. 理解の深さをコントロールする
   - 「前から順番にすべてを完璧に理解していき，理解できないところに直面した際には立ち止まって，悩みすぎる」のではなく，「まずは浅く全体像を理解して，再度読み直す」
2. 無知の知
   - 自分が勉強をしていて，何を理解していて，何を理解できていないのかを明確にすることが大切．
   - たとえば，機械学習のアルゴリズムにおいて，各専門用語の意味や役割をきちんと自分の言葉で説明できるか，各式に出てくる記号や各項を理解できているか，式が意味するところを言葉として説明できるか，式変形のしかたを説明できるか．
3. 他人が実装したコードを見ること
4. アウトプット駆動学習
   - 自分が機械学習で趣味的に何か実現したいこと（アウトプット）を決めて，それをいろいろなアルゴリズムで実装し，その過程で必要になった知識を順次勉強していく．