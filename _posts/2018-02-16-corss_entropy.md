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


$$H(x\mid y) = E_{X, Y}[log(p_X(x \mid y))]$$

$$p_X(x \mid y) = {p_{X,Y}(x,y)\over p_Y(y)},  \text{beyes' rule}$$

여기서 $$q(x,y) = log(p_X(x \mid y))$$로 치환에서 대입을 하면

$$E_{X, Y}[log(p_X(x \mid y))] = \int_{X} \int_{Y} p_{X, Y}(x,y)log(p_X(x\mid y))dydx$$

    치환해서 대입하는 것은 맞았지만 치환해야 하는 대상이 틀렸었다. Conditional entropy를 정의 하기 위해서는 entropy식에서 바로 확장하는게 아니라 entropy의 근본적인 개념 즉, 평균을 구하는 식을 가지고 Conditional entropy를 정의 했었어야 했다. 좀 더 확장해서 생각해보면 기초가 튼튼하지 않으면 새로운 알고리즘을 만드는 건 어려운게 아닐까 생각한다.  


#### KL divergence

#### cross entropy

#### Mutual information

### 결론
