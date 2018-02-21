---
title: 딥러닝 논문을 읽을 때 알면 좋은 것들?(Entropy와 관련된 기호)(미완)
description:
categories:
 - study
tags:
 - probability
---

딥러닝 논문을 읽다보면서 가장 어려웠었던 건 수학 기호들이 이해가 안 되는건 물론이고 논문마다 수학 기호가 조금씩 다르게? 사용을 했었다. 그 이유는 [리처드 파인만의 저서인 물리법칙의 특성](https://www.goodreads.com/book/show/27228256)에서 설명을 아래와 같이 했다.

> 물리학자들은 그토록 다양한 방식으로 에너지를 도입하고 또한 다른 단위로 측정하고 다양한 이름으로 명명한 것을 부끄러워해야만 한다. 모든 것이 다 정확히 똑같은 것을 측정하는 단위임에도 불구하고, 에너지를 칼로리, 에그르, 전자볼트, 피트 파운드, 마력 시간, 킬로와트 시간등으로 잰다는 것은 우스꽝스러운 일이다. (생략). 물리학자들이 인간이라는 것을 확인하고 싶은 사람이 있다면, 물리학자들이 에너지를 나타내기 위해 이 모든 다양한 단위들을 사용하는 어리석음을 범하고 있다는 사실이 좋은 증명이 될 수 있을 것이다.

즉, 생각보다 그들은 대단히 뛰어나고 현명하지 않다고 리처드 파인만은 생각 했던 것 같다.


### 정보이론에서 나오는 Entropy 관련 수식


>$$x \sim p_{X}$$ 의 의미는 확률 변수 $$x$$는 $$p_{X}$$의 분포를 따른다.

이 말을 좀 더 자세히 이해해 보면, 여기서 $$x$$는 간단하게 어떤 이미지 데이터라고 생각하면 된다.

 $$p_{X}$$의 분포를 따른다는 말은 일단 먼저 $$p_{X}$$가 의미하는 것을 알아야 한다.

보통 대문자 $$X$$는 다음과 같이 $$X = \{x_1,x_2,x_3,...,x_n\}$$ 어떤 집합을 의미하고, 여기서 $$p_X$$는 $$X$$의 원소를 입력으로 하는 함수라고 생각하면 된다. 다르게 말하면 $$X$$의 원소가 아닌 것은 함수의 변수로 사용하지 않는다고 생각하면 된다.

여기서 함수 $$p_X$$역할은 [Probability Density Function(PDF)](https://en.wikipedia.org/wiki/Probability_density_function)이다. 가끔 논문에서 Probability Distribution Function이라고 쓰는것 같다. 비슷한 말로는 [Cumulative Distribution Function(CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function)가 있는데, 처음에는 Probability Distribution Function를 CDF로 생각 했었는데, 수식을 이해하다보면 CDF가 아니라 PDF로 쓰인 것 같다. 따라서 아래에서 쓰는 함수들은 대부분 PDF이다.


#### Entropy




$$H(x) = -\int p(x)log(p(x))dx$$

$$H(x) = -\int_{X} p_{X}(x)log(p_{X}(x))dx$$

$$H(x) = - E_{x \sim p_{X}(x)}[log(p_{X}(x))]$$

위의 식은 모두 똑같은 entropy식이다.(continuous 경우만 고려 했다.) 자세한 설명은 아래 블로그를 참조하면 좋다.

 [https://ratsgo.github.io/statistics/2017/09/22/information/](https://ratsgo.github.io/statistics/2017/09/22/information/)

[http://sanghyukchun.github.io/62/](http://sanghyukchun.github.io/62/)

PDF함수를 간단하게 표현하면 그냥 $$p(x)$$로 쓸 수도 있고, 만약 PDF함수가 여러게면 $$p_X(x)$$ or $$p_Y(y)$$게 표현 할 수 있다. 하지만 꼭  $$p_X(x)$$에서 $$X$$(대문자)를 사용했다고 변수를 반듯이 $$x$$(소문자)로 사용할 필요가 없다. 개인적으로는 이래서 논문을 이해할 때 많이 헷갈린다. [Generative Adversarial Networks(2014)](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)논문에서 나온 예를 보이면,

 ![](/assets/2018-02-12/2.png)


$$p_{data}(x)$$라는 함수를  해석해 보면 $$x \sim p_{data}$$의미로 해석되고 $$p_{data}$$는  $$x$$를 변수로 갖는 함수이다.  

그리고 논문을 자세히 읽어보면 $$x \sim p_{data}$$라는 언급이 아래에 문장에서 나온게 아닐까 생각한다.

> We also define a second multilayer perceptron D(x; θ<sub>d</sub>) that outputs a single scalar. D(x) represents the probability that x came from the data rather than p<sub>g</sub>.


또 논문을 읽다 보면, 수식이 아래와 같이 자주 바뀌면서 설명하는 것을 볼 수 있다.

$$\int_{X} p_{X}(x)log(p_{X}(x))dx =  E_{x \sim p_{X}(x)}[log(p_{X}(x))]$$

 ![](/assets/2018-02-12/3.png)

이 그림의 위의 그림과 바로 이어지는 부분인데, 바로 수식을 바꿔서 설명한 것을 알 수 있다.

시작은 entropy식을 좀 더 융통성 있게 보면 결국  $$-log(p(x))$$의 평균이라고 생각하면 된다. 그리고 위에서 사용한 평균 기호를 좀 더 일반화 하면
$$E_{x \sim p_{X}}[q(x)] = \int_{X} p_{X}(x)q(x)dx$$
와 같이 변하는 것 같다.

#### 확률에서의 평균이란?

$$E_{x \sim p_{X}}[q(x)] = \int_{X} p_{X}(x)q(x)dx$$

위의 식이 왜 평균을 구하는 식인지 직관적으로 이해해보자.
사실 내가 처음에 이해 했던 평균은 다음과 같다.

$$average = {1\over n} \underset{x=1}{\overset{n}\sum}x$$

위의 식은 1부터 n까지의 합이다. 여기서 한 단계 더 일반화를 해보자. 1부터 n까지의 합이 아니라 $$y = 2\sin(x)+3$$ 함수에 $$x$$는 1부터 n까지를 각각 대입한 결과값 $$y$$의 평균을 구한다고 생각해보자 그러면 식은 아래와 같아 진다.

$$average = {1\over n} \underset{x=1}{\overset{n}\sum}y$$

위의 식은 잘못된 표현이다. 왜냐하면 $$y$$의 값이 $$x$$에 따라 변한다는 표현이 없기 때문이다. 따라서 $$y = 2\sin(x)+3$$를 $$y = q(x)$$로 정의하고 대입하면,

$$average = {1\over n} \underset{x=1}{\overset{n}\sum}q(x)$$

사실 함수의 개념을 좀 더 확장해보면, $$q(x)$$는 아무거나 다 될 수 있다. 예를 들어 딥러닝에서는 $$x$$가 이미지 데이터이고 $$q(\cdot)$$은 뉴런 네트워크이고, $$y$$값은 feature map일 수도 있고, edge 이미지일 수 도 있다.

    이부분은 사람들이 보통 지식으로 암기하고 있다. 암기하고 있으면 사람들은 보통 안다고 하지만 개인적인 무언가를 안다는 것은 암기한 것을 응용을 할 수 있어야 안다고 말할 수 있다고 자신있게 말할 수 있는 것 같다.

만약 $$n = 4$$라고 하면

$$average = {q(1) + q(2) + q(3) + q(4)\over 4}$$

단순히 시그마 기호를 없앴다. 여기서 만약 여기서 $$q(2)$$와 $$q(4)$$가 각각 1개와 2개씩 추가된 상태에서의 평균을 구하면 아래와 같아진다.


$$average = {q(1) + 2q(2) + q(3) + 3q(4)\over 7}$$

밑의 수가 4에서 7로 바뀐 이유는 총 개수가 3개 추가 됐기 때문이다. 만약 위의 식을 다시 시그마가 있는 식으로 바꾸려면 어떻게 해야할까? 쉬운 방법은 함수를 아래와 같이 정의 하는 것이다.

$$p(1) = {1\over 7}, p(2) = {2\over 7}, p(3) = {1\over 7},p(4) = {3\over 7} $$

위에서 정의한 식을 다시 대입해서 표현하면,

$$average = p(1)q(1) + p(2)q(2) + p(3)q(3) + p(4)q(4)$$

와 같이 변하게 되고 시그마로 정리하면

$$average = \underset{x=1}{\overset{4}\sum}p(x)q(x)$$

여기서 $$p(x)$$는 우리가 생각하는 [Probability Density Function(PDF)](https://en.wikipedia.org/wiki/Probability_density_function)이다. 위의 수식의 의미를 해석해 보면 $$p(x)$$는 $$q(x)$$라는 특정 값의 확률이라고 생각하면 된다. 물론 이렇게 해석할 수 있는 것은 discrete 버전이기 때문이고 continue 버전에서는 어떤 특정 값의 확률은 무조건 0이기 때문에 범위 개념으로 잡아서 확률을 계산한다. 어쨌든 discrete이든 continue이든 개념은 거의 같기 때문에 위와 같이 이해하면 될 것 같다.

    사실 PDF의 개념을 하나의 문장으로 설명하는 것은 진짜로 이해하는 데 아무런 도움이 안되는 것 같다. 왜냐하면 하나의 문장으로 PDF의 개념을 내 능력으로는 못 쓰겠다. 그래서 위와 같이 나름데로 스토리로 만들어서 이해하고 있다.

위의 식을 continue 버전으로 바꾸면

$$average = \int p(x)q(x)dx$$

로 바뀌게 된다. 물론 $$p(x)$$를  $$p_X(x)$$ 또는  $$p_{data}(x)$$와 같이 쓸 수 있다.

    이정도 설명이면 나름 확률에서 평균이라는 식의 의미를 엄청 대충 감이라도 잡을 수 있지 않을까 생각한다.

#### Conditional entropy

$$H(x\mid y) = -\int \int p(x,y)log(p(x\mid y))dy dx$$


[infoGAN](https://arxiv.org/abs/1606.03657) 논문을 이해하기 위해서는 Conditional entropy를 알아야 한다. 왜냐하면 Mutual information식에 Conditional entropy식이 있기 때문이다.

처음 Conditional entropy를 아래와 같이 이해했다.

$$H(x\mid y) = - \int p(x\mid y)log(p(x\mid y))dx$$

위과 같이 생각한 이유는 단순히 아래의 식에서

$$H(x) = - \int p(x)log(p(x))dx$$


~~$$x$$를 $$x\mid y$$ 로 치환한 것 뿐이였다. 하지만 잘 생각해 보면 이건 논리적으로 안 될수도 있다고 생각한다. 치환이라는 것은 $$y=q(x)$$와 같이 등식이 있는 것에 적용될 수 있는데, 여기서는 $$x = x\mid y$$가 성립이 안된다. 확률에서는 변수를 보통 사건이라고 부르는데 여기서 $$x$$와 $$c$$는 서로 다른 사건이다. 왜냐하면 조건부 확률은  서로 다른 사건과의 관계를 나타내는 확률? 이기 때문이다. 이 말을 변수로 생각하면 $$x$$와 $$c$$는 서로 다른 변수이다. 따라서 $$x = x\mid y$$ 처음부터 말도 안되는 생각이 였다. 사실 살면서 $$x = x\mid y$$ 이런 등식을 본적이 없기도 했다.~~ (아래는 보면 알겠지만 이 생각은 반은 맞고 반은 틀렸었다.)

이전까지는 하나의 변수에 대한 평균에 대해서만 고려 했었다. Conditional entropy를 정의 하기 위해서는 두 개의 변수가지는 평균식의 정의를 알아야 한다.

$$E_{x \sim p_{X}, y \sim p_{Y}}[q(x,y)] = \int_{X} \int_{Y} p_{x \sim p_{X}, y \sim p_{Y}}(x,y)q(x,y)dydx$$

확률에서 보통 대문자와 소문자의 관계는 $$x \in X $$와 같다. 이 의미는 $$x$$의 모든 값은 $$X$$에 속한다는 말이다. 따라서

$$x \sim p_{X}, y \sim p_{Y} \rightarrow  X, Y$$

로 표현되기도 한다. 그래서 식을 다시 써보면,


$$E_{X, Y}[q(x,y)] = \int_{X} \int_{Y} p_{X, Y}(x,y)q(x,y)dydx$$

위의 식은 하나의 변수로 이루어진 평균식과 비교하면 개념을 쉽게 확장할 수 있다. 이제 위의 식을 가지고 Conditional entropy를 정의하면 아래와 같다.


$$H(x\mid y) = E_{X, Y}[-log(p_X(x \mid y))]$$

$$p_X(x \mid y) = {p_{X,Y}(x,y)\over p_Y(y)},  \text{beyes' rule}$$

여기서 $$q(x,y) = -log(p_X(x \mid y))$$로 치환에서 대입을 하면

$$E_{X, Y}[-log(p_X(x \mid y))] = -\int_{X} \int_{Y} p_{X, Y}(x,y)log(p_X(x\mid y))dydx$$

    치환해서 대입하는 것은 맞았지만 치환해야 하는 대상이 틀렸었다. Conditional entropy를 정의 하기 위해서는 entropy식에서 바로 확장하는게 아니라 entropy의 근본적인 개념 즉, 평균을 구하는 식을 가지고 Conditional entropy를 정의 했었어야 했다. 좀 더 확장해서 생각해보면 기초가 튼튼하지 않으면 새로운 알고리즘을 만드는 건 어려운게 아닐까 생각한다.  





#### KL divergence

$$KL(p\| q) = D_{KL}(p\| q) = -\int p(x)log({q(x)\over p(x)})dx$$

위의 식은 인터넷에서 검색만 하면 아는 KL divergence 식이다. 보통 이 식은 두 확률 분포의 차이를 구할 때 쓴다고 한다. 그럼 이제 위의 식을 분해해서 왜 두 확률분포함수의 차이를 구하는지 이해해보자.

$$-\int p(x)log({q(x)\over p(x)})dx = -\int p(x)log(q(x))dx - (-\int p(x)log(p(x))dx)$$

log의 성질을 이용하면 위과 같이 바뀐다. 위 식을 평균의 식에 대입해보면

$$-\int p(x)log({q(x)\over p(x)})dx = E_{x \sim \color{blue}{p(x)}}[-log( \color{red}{q(x) })] - E_{x \sim \color{blue}{p(x)}}[-log(\color{blue}{p(x)})]$$

위의 식을 보면 이상한 점이 있다.(이상하기 생각하는 부분을 blue와 red로 표현 했다.) 이 식은 보통 두 확률분포의 차이를 구할 때 사용한다고 하지만 위의 식을 통해서 분석해 보면 하나의 확률분포에서 나오는 서로 다른 두 가지 값의 차이를 계산하는 것 처럼 보인다. 내가 생각했던 두 확률 분포의 차이는 다음과 같았다.

$$E_{x \sim \color{red}{q(x)}}[-log(\color{red}{q(x)})] - E_{x \sim \color{blue}{p(x)}}[-log(\color{blue}{p(x)})]$$

사실 내가 생각한 식을 보면 단순히 평균값의 차이이다. 평균값이 서로 같다고 해서 두 값은 서로 같다고 말하기는 좀 부정확하다. 간단하게 예를 든다고 해도 가우시안 분포를 따르는 확률분포함수도 평균이 같아도 분산이 다르면 서로 다르다고 말해야 하는게 아닌가? 하지만 위의 식으로는 평균은 같기 때문에 같다고 생각하는 오류가 발생한다. 사실 이건 평균값의 한계인 것 같다. 왜 위와 같이 정의됬는지 이해하기 위해서는 왜 만들어졌는지 알아야 하는 것 같다.



>In information theory, the Kraft–McMillan theorem establishes that any directly decodable coding scheme for coding a message to identify one value $$x_i$$ out of a set of possibilities $$X$$ can be seen as representing an implicit probability distribution $$q(x_i)=2^{-l_i}$$ over $$X$$, where $$l_i$$ is the length of the code for $$x_i$$ in bits. Therefore, the Kullback–Leibler divergence can be interpreted as the expected extra message-length per datum that must be communicated if a code that is optimal for a given (wrong) distribution $$Q$$ is used, compared to using a code based on the true distribution $$P$$

위의 글은 [위키피디아 motivation](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) 부분이다. 이 부분을 읽어보면 $$q(x_i)=2^{-l_i}$$이 부분을 wrong distribution $$Q$$로 정의 했다. 이 부분이 왜 확률분포함수로 바뀌는지 모르겠다. 그래서 나름 관련된 자료를 읽고 왜 이렇게 됬는지 예상을 해보았다.

관련 자료 : [Entropy encoding](https://prateekvjoshi.com/2014/12/06/what-is-entropy-coding/), [허프만 알고리즘](http://wooyaggo.tistory.com/95). [Kraft–McMillan inequality](https://en.wikipedia.org/wiki/Kraft–McMillan_inequality)

엔트로피는 과거 효율적으로 데이터를 전송하기 위해서 만들어진 걸로 알고 있다. 현재(2018년)에는 모르겠지만 과거 하드웨어 기술이 지금만큼 발전하지 못해서 어떤 정보(암호나 문장 같은 것)를 전송하기 위해서는 최소한의 데이터 길이로 보내야 했다. 하지만 사람들이 쓰는 정보는 보통 문장이였는 데 그 문장 내용에 따라 알파벳이 달랐고, 그 각각의 알파벳의 빈도 수가 달랐다. 위의 관련자료를 다 읽어보면 알겠지만, 각각의 알파벳을 일정한 크기의 bit(디지털 신호는 0과 1의 조합이기 때문에)로 정의 하는게 아니라 빈도 수가 많은 것을 짧은 bit(정보량이 작음)로 빈도 수가 적은 것은 긴 bit(정보량이 큼)로 표현해서 데이터를 보내면 모든 알파벳을 같은 bit의 크기로 정의하는 것보다 평균적으로 더 짧은 bit의 크기로 정보를 전송할 수 있다. 이것을 수학적으로 표현하면 [정보 엔트로피](https://ko.wikipedia.org/wiki/정보_엔트로피)이다.

$$H(x) = E_{x \sim p_{X}(x)}[-log(p_{X}(x))]$$

하지만 위의 식에 문제가 있는데 위에서 정의한 정보량 $$-log(p_{X}(x))$$의 값을 어떻게 정의 하느냐이다. 그 정의하는 방법 중 하나는 아마도 [Entropy encoding](https://prateekvjoshi.com/2014/12/06/what-is-entropy-coding/)과 관련 있고, 실제 정보량을 정의하기 위해서는 반듯이 정수로 나와야 한다.

그래서 나온 수식이 $$q(x_i)=2^{-l_i}$$이고 이것은 아마도 [허프만 알고리즘](http://wooyaggo.tistory.com/95)으로 정의한 것의 정보량을 표현하는 식의 결과일 것이다. 운이 좋게도 위의 식의 합은 [Kraft–McMillan inequality](https://en.wikipedia.org/wiki/Kraft–McMillan_inequality)식에 의해서 최댓값은 1이다. 이것은 확률분포함수로 정의하기에 충분한 조건이다.

알고리즘([허프만 알고리즘](http://wooyaggo.tistory.com/95)과 같은)을 개발하면 성능의 효율성을 보이기 위해서 적절한 measure 방법이 필요하고, 그것이 $$KL(p\|q)$$게 아닐까 생각한다.

즉, $$KL(p\|q)$$는 사람들이 정의한 정보량($$-log(\color{red}{q(x)})$$)을 가지고 실제 확률분포($$\color{blue}{p(x)}$$)를 이용한 평균값이  실제 확률분포($$\color{blue}{p(x)}$$)로 정의한 엔트로피의 차이를 구한 measure 식이다.

$$KL(p\|q) = E_{x \sim \color{blue}{p(x)}}[-log( \color{red}{q(x) })] - E_{x \sim \color{blue}{p(x)}}[-log(\color{blue}{p(x)})] \ge 0$$

$$E_{x \sim \color{blue}{p(x)}}[-log( \color{red}{q(x) })]  \ge 0$$

$$ E_{x \sim \color{blue}{p(x)}}[-log(\color{blue}{p(x)})] \ge 0$$

$$KL(p\|q) \neq KL(q\|p)$$  

처음에는 단순히 정보를 효율적으로 보내는 알고리즘을 성능을 측정하기 위해 사용됬을 거라고 생각하나, 나중에 두 확률분포가 얼마나 다른지도 사용할 수 있는 수학적 근거가 명확해 져서 현재 $$KL(p\|q)$$는 두 확룰분포함수를 비교할 때 쓴다고 말하는게 아닐까 생각한다.




#### cross entropy

$$KL(p\|q) = E_{x \sim \color{blue}{p(x)}}[-log( \color{red}{q(x) })] - E_{x \sim \color{blue}{p(x)}}[-log(\color{blue}{p(x)})]$$

cross-entropy식의 경우 $$KL(p\|q)$$식에서 $$E_{x \sim \color{blue}{p(x)}}[-log( \color{red}{q(x) })]$$ 부분만 가져온 식이다. 이 식은 딥러닝에서 loss함수로 많이 사용된다. 딥러닝에서 많이 쓰는 이유는 [sigmoid 함수](https://en.wikipedia.org/wiki/Sigmoid_function)와 같이 썼을 경우 [Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error)에 생기는 gradient vanishing 문제를 일부분 해결할 수 있기 때문이다.

    딥러닝에서 실제 쓰는 loss 함는 엄밀하게 말하면 KL divergence가 아닐까 생각한다. 하지만 미분을 하면 cross entropy 식만 남기 때문에 실질적으로 의미가 없는 부분을 제거하고 사용한게 아닐까 생각한다. 그래서 처음 딥러닝 공부를 할 때, cross entropy 식이 뜬금?없이 나와서 이상했었는 데 알고보니 두 확률값의 차이를 구하기 위해 정의한 식의 생략 버진이라고 생각하니 뭔가 받아들이기 쉬워졌다.



#### Mutual information

$$I(x,y) = KL(p_{X,Y}(x,y)\| p_X(x)p_Y(y))$$

이 수식은 두 확률분포가 독립(independent)인지 종속(dependent)인지에 관한 식이다. 만약 어떤 두 확률 분포가 독립이라면 아래의 식이 만족한다.

$$p_{X,Y}(x,y) = p_X(x)p_Y(y)$$

독립사건과 종속사건에 대한 자세한 설명은 [이곳](http://j1w2k3.tistory.com/773)을 참고하면 좋을 것 같다.

막상 Mutual information이 독립사건이면 결과 값이 0 아니면 0보다 큰 값이라고 나오는 수식이라고 이해하니 더 이상 궁금한게 없어졌다. 이 부분의 내용은 나중에 [infoGAN](https://arxiv.org/abs/1606.03657) 논문을 review하면서 좀 더 자세히 설명할 기회가 있을 거라고 생각한다.




### 결론

[infoGAN](https://arxiv.org/abs/1606.03657) 논문을 읽다가 Mutual information이 뭔지 알아보려고 공부하다가 쓴 포스트 이지만 생각보다 오랜 시간이 걸렸다.

이 포스트를 보면 처음에는 모르는게 많아서 어떤 기초개념?을 알기 위해서 부수적인 것들이 많았지만 어느 정도 익히고 난 후에는 새로운 개념을 익히는게 수월해졌다.

사실 이런거 몰라도 github에서 소스코드 다운받은 후 코드를 분석하는게 알고리즘을 이해?하는게 훨씬 빠른 방법일 것이다. 대부분의 사람들은 이 방법을 추천할 것이다. 이 방법의 장점은 결과를 빠르고 쉽게 얻을 수 있는 것이다. 하지만 ...
