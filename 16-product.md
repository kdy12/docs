# 产品设计与推广

## 💡 这节课会带给你

1. 如何详细设计产品
2. 如何快速验证想法
3. 如何让更多人知道你的产品

开始上课！


当已经定义好产品的方向和商业模式，接下来就可以设计产品了。


## 产品设计


简单说，就三件事：

1. 功能设计
2. 信息架构设计
3. 交互设计

如果对视觉要求高，还需要特别做视觉设计。否则套模板就差不多。

对全栈来说，三件事不分先后，而是彼此交错。

但起手做的，常常是：抄。


### 怎样优雅地~~抄~~借鉴产品


借鉴的合理性：

1. 不该打破用户习惯
2. 这其实是行业惯例了

优雅的姿势：

1. 先自己设计，然后再参考
2. 学结构、学交互，别学视觉
3. 取各家之长做整合
4. 功能抽象。做一个简单的功能解决很多问题，而不是做很多功能各解决各自的问题

避坑关键：

1. 用 5 why 法推演为什么人家这么设计
2. 如果找不到理性答案，慎抄，容易进坑


### 写 PRD


一定要先写文档！俗称 PRD（Product Requirement Document）。

1. 虽然可能写完没人看，但写文档的过程比结果更重要，可以让思路更清晰
2. 飞书的「[产品需求文档](PRD.docx)」模板特别好用，往里填就行
3. 可以用 ChatGPT 找些思路和灵感，常见功能它写得还不错
4. 试试在 IDE 里开着 Copilot 写文档，体验很棒（「体验很棒」这四个字，就是 Copilot 写的）

![copilot](copilot.png)


### 生成式 AI 产品设计原则

在设计产品的过程中，时不时检查，是否这些原则全部符合。

#### GitHub Copilot 总结的四个关键支柱

GitHub Copilot 是最早，也是迄今为止最成功的基于大模型 AI 的垂类工具产品。其经验非常值得借鉴。

Copilot 团队也乐于分享。在 [How we’re experimenting with LLMs to evolve GitHub Copilot](https://github.blog/2023-12-06-how-were-experimenting-with-llms-to-evolve-github-copilot/) 一文中，总结了 GitHub 设计 AI 产品的四个关键支柱（Key Pillars）：

> - **可预测**。我们希望创建能引导开发人员实现最终目标的工具，但不会让他们感到惊讶或不知所措。
> - **可容错**。正如我们所见，AI 模型可能犯错。用户应该能够轻松地发现不正确的建议，并以较低的成本来解决，以便专注和提高生产力。
> - **可操控**。当回答不正确或不符合用户期望时，他们应该能够引导 AI 找到解决方案。否则，我们就是乐观地指望模型产生完美的答案。
> - **可验证**。解决方案必须易于评估。模型并不完美，但如果用户能验证其输出，它们就能成为非常有用的工具。


#### 补充一个原则：有反馈

AI 生成的内容，用户认可还是不认可，需要有反馈。

这个反馈可以是：

1. 用户主动上报，比如 ChatGPT 的 👍 和 👎
2. 用户的行为，比如 Copilot 里是否按了 `tab` 键，ChatGPT 里是否点击了 copy 按钮


### 画原型图


编程背景的会更喜欢直接上代码，省却画原型步骤。

代码写出来就有感情，往往不爱改。所以，先画原型更有助于迭代，能在具象和整体层面上，更好地把控。

和传统流程不同，我们原型图不需要太精细，线框图足够了。细节反倒是在代码里做调整更直接、准确。

推荐的工具：

1. [draw.io](https://draw.io) - 免费开源，对程序员最友好的画图工具，可以直接改 style。内置的图形库，几乎就足够用了
2. [Figma](https://www.figma.com) - 专业的原型设计工具，非常非常强大，但学习成本高，建议处理 logo、icon 等图片切图时使用。免费版就够用。收购了一家 AI 公司，未来会有更多 AI 功能
3. [Motiff](https://motiff.com/) - 国产新秀，还在测试阶段。AI native 的设计工具
4. [v0](https://v0.dev/) - 从文字生成设计图，用对话调整设计图，直接生成代码


### draw.io

1. 可以直接编辑云文件，包括 GitHub
2. 保存的 .png 和 .svg 文件还能继续编辑
3. Bootstrap、Mockup、Android、iOS 等图形库，非常全面
4. 容器功能特别好用，画架构图首选
5. 有客户端，速度更快


### Figma

1. 把 sketch 按到地上摩擦的专业设计工具
2. 可以写 plugin 来扩展其功能
3. 适合多人协作，支持实时协同编辑
4. 直接生成样式代码
5. 有客户端，速度更快


### Motiff

1. 基本编辑功能和 Sketch、Figma 大同小异
2. AI 魔法框能力很独特，堪称设计界的 Copilot
3. AI 识别通用组件也挺有用


### v0

无需多言，用就知道了。


### 极简 UI 设计原则

1. 所有元素要**对齐**。和保持代码缩进一样重要
2. **亲密性**，相关元素离得更近。和代码里加空行一个道理
3. 同一界面**最多使用 4 种颜色**。颜色的意义：传递信息 > 美观
4. 用 **4 的倍数**（4、8、12、16、20……）设置大小、间距。视觉感受不明显，但适配性更好

<img src="design_principles.png" width="600px" />

推荐[《写给大家看的设计书（第 4 版）》](https://www.dedao.cn/ebook/detail?id=rJRdy1qe4xAVBgZrvdGmz8ykaop6QWXnEEwEJnD7LR51qb2KY9NPMXOljVa28m5K)。我用半本书就撑住了半辈子的设计工作……


### AI 产品的典型界面风格：Collaborative UX

微软在开发 Copilot 时提出「Collaborative UX」，协作式用户体验。

目前还没有更详细、系统的总结，只是一个理念。

<div class="alert alert-success">
<strong>Collaborative UX 的核心思想：</strong>把 AI 当人看，想象成人在一起操作，界面怎样最好用？
</div>

几种常见风格：

1. IM 对话：ChatGPT、Bing Chat、Copilot Chat
2. 同文档协作：GitHub Copilot、各种在线协作工具
3. 对话+协作：把对话嵌入到协作点，模拟两个人一起在屏幕前指指点点。比如 GitHub Copilot 在代码中按 `Cmd + I`


## 快速搭 demo

可以有个期待，AI 把 PRD 直接变成产品。[MetaGPT](https://github.com/geekan/MetaGPT) 在这方面走得最远，但还很不够。

懂代码的我们，可以用 [Gradio](https://www.gradio.app/)，快速搭出一个主流程 demo，供用户/客户体验，验证需求。


### Gradio


https://www.gradio.app/

1. 是 AI 界最常用的界面开发库
2. 非常简单，用纯 python 就能写出功能强大的 Web 界面，不需要任何 JavaScript、CSS 知识
3. 响应式设计，支持手机、平板、电脑等多种设备
4. 文档详细，样例丰富，非常容易上手

比如：

- Chatbot Arena: https://chat.lmsys.org/
- Stable Diffusion web UI: https://github.com/AUTOMATIC1111/stable-diffusion-webui

安装：

```bash
pip install gradio
```


### Hello world

先从一个最简单的例子开始，看看 Gradio 的基本用法。

[hello-app.py](hello-app.py)


### 仿 ChatGPT

用 Gradio，快速搭出一个 ChatGPT。

[chat-app.py](chat-app.py)


### 更高级的 ChatGPT

加上更多参数，适合专业人士使用。

[chat-pro-app.py](chat-pro-app.py)


### 高级定制界面

前面演示的都是用内置模板直接创建界面。但 Gradio 还支持高级自定义界面。

界面中可以放置的组件，有：

- AnnotatedImage
- Audio
- BarPlot
- Button
- Chatbot
- Checkbox
- CheckboxGroup
- ClearButton
- Code
- ColorPicker
- Dataframe
- Dataset
- Dropdown
- DuplicateButton
- File
- Gallery
- HTML
- HighlightedText
- Image
- Interpretation
- JSON
- Label
- LinePlot
- LoginButton
- LogoutButton
- Markdown
- Model3D
- Number
- Plot
- Radio
- ScatterPlot
- Slider
- State
- Textbox
- Timeseries
- UploadButton
- Video

界面布局用 tab、row、column 模式，例如：[layout-app.py](layout-app.py)


## 开发

略

## 产品运营


<div class="alert alert-warning">
<b>焦点讨论</b>： AI 创世宫斗中，为什么 OpenAI 选择 Sam Altman，而不是 Ilya Sustkever？
<ol>
<li>Sam Altman：前 OpenAI CEO</li>
<li>Ilya Sustkever：OpenAI 首席科学家</li>
</ol>
</div>


运营好于产品的例子（他们的对手就是反例）：

1. 海底捞
2. 瑞幸咖啡
3. ofo
4. 中国云计算的 top 厂商

产品如果没有压倒性优势（比如 ChatGPT），从一开始就是拼运营。

只要市场足够肥沃，压倒性优势的产品，长期一定会被竞争对手追上，进入拼运营的阶段。

<div class="alert alert-success">
<b>划重点：</b>长期看，运营比研发重要很多。
</div>

运营核心三件事：

1. 获客
2. 转化
3. 留存

但运营是个具体而微的工作，无数细节，无数变化。比调 prompt 折腾多了，因为面对的是真正的人。


### 获客

#### 获客核心

<div class="alert alert-success">
把最打动人的宣传，呈现给最有可能成为用户的人。
</div>

宣传和产品解决的「真需求」是一脉相承的。**能否一句话讲清产品是做什么的**，有时就是成功和失败的分水岭。

- 美团外卖，送啥都快
- 多快好省逛京东
- 知识就在得到
- 知乎：有问题，就会有答案
- 米堆学堂，生活向上

注意：宣传语（slogon）押韵很重要，好记，且有神奇的说服力。

#### 免费/低价获客

1. 口碑传播
   - 价值最高的获客方式
   - 用产品自身说话，低成本获客，形成正向循环
   - 需要时间积累，起速慢
   - 不容易突破圈层
2. 社交媒体推广，用内容获客
   - 知乎、公众号、视频号、抖音、快手、B 站、小红书、X、TikTok、Facebook 等
   - 初期效果不明显，但长期坚持会越来越好
   - 根据受众选择媒介
   - 用好媒介的推荐策略
3. SEO
   - 老派，但仍有用
   - 尤其适合 2B
4. 开源
   - 非常利于免费传播
   - 开源也是一种商业策略
5. 独立开发者友好的社区
   - [v2ex](https://www.v2ex.com/)
   - [即刻](https://web.okjike.com/)
   - [Product Hunt](https://www.producthunt.com/)

#### 付费获客

谨慎花钱！

1. 专业渠道
   - 有些渠道公司专门做对应市场的推广，比如高校渠道、HR 渠道等
2. 线下渠道
   - 到目标用户集中出现的地方
   - 学校、商场、展会等
3. 意见领袖
   - 各种大 V、达人，背书效果好
4. 广告平台
   - 朋友圈、微博、抖音、快手等
   - 流量巨大，几无上限
   - 线下广告牌、电视广告等（特别烧钱）


### 转化

1. 如果产品强/客单价低，能形成自然转化
2. 如果产品弱/客单价高，要依赖销售策略（话术）

客单价越高，对销售的依赖越大。

<div class="alert alert-success">
销售过程的本质是：挖需 + 针对性解决方案 + 创造超预期的体验
</div>

此处挖需，千万别用 5 why 法，但要有 5 why 法的精神。


### 留存

<div class="alert alert-success">
保持接触是留存的关键。
</div>

1. 高频使用的产品比低频更容易留存，所以如果可能，优先做高频产品
   1. 案例：微信 vs. 支付宝
   2. 反例：妙鸭
1. 产品价值和服务，是留存的核心
1. 通过社交媒体保持触达，把留存和获客合起来做
1. 建私域，做私域运营


## 迭代

<div class="alert alert-success">
<b>划重点：</b>谨慎增加新功能，保持小而美
</div>

1. 别用高频迭代代表勤奋，要用积极运营代表勤奋
2. 第一版只做主路径功能。但别忽略细节体验
3. 用 KANO 模型，找到提升**核心用户**满意度的功能，只做这些
   - 非核心用户骂你，也不要理睬
   - 别轻易改变**核心用户**画像。如果要变，可能做个新产品更好

![kano](kano.webp)

<div class="alert alert-warning">
<b>思考：</b>兴奋型、期望型、基本型需求，怎样的优先级？
</div>


## 扩展学习

1. [《邱岳·互联网产品方法论》](https://www.dedao.cn/course/detail?id=Ox1El850jp9VaZasnvJZg6MbrdvRBo) - 最真实的
2. [《产品思维 - 从新手到资深产品经理》](https://www.dedao.cn/ebook/detail?id=nroX7MYDaKMjy7eNqrmOX6pnAQ5Vg049x2WJzxbE9LZl1o8RkGd2BPYv4x6d9meB) - 逻辑最严密的，适合程序员
3. [《运营之光：我的互联网运营方法论与自白 3.0》](https://www.dedao.cn/ebook/detail?id=bxEYR1zAbZMmVzK4p1oxl67XeNaB83OqQV092GJERgryYQdDnqjkPLvO5eOZ8Nqr) - 最久经考验的

