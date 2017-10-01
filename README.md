# Assignments
Assignments for newbies (students of Prof. Gao)

<h2>Study Plan 学习计划</h2>
<p>以下为针对2017-2018学年研一及保研同学的《统计学习方法》一书的阅读进度安排。</p>
<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th width="15%" scope="col"> 章节 </th>
    <th width="55%" scope="col"> 内容 </th>
    <th scope="col"> 日期 </th>
    <th scope="col" class="text-center"> 负责人 </th>
    </tr>
  <tr>
    <td>第1章</td>
    <td>统计学习方法概论1-24页</td>
    <td>10.09-10.15</td>
    <td> </td>
    </tr> 
     <tr>
    <td>第2章</td>
    <td>感知机25-36页</td>
    <td>10.09-10.15</td>
    <td>  </td>
    </tr>
     <tr>
    <td>第3章</td>
    <td>k近邻法37-45页</td>
    <td>10.16-10.22</td>
    <td> </td>
    </tr> 
     <tr>
    <td>第4章</td>
    <td>朴素贝叶斯法47-53页</td>
    <td>10.16-10.22</td>
    <td>  </td>
    </tr>
        <tr>
    <td>第5章</td>
    <td>决策树55-75页</td>
    <td>10.23-10.29</td>
    <td>  </td>
    </tr>
     <tr>
    <td>第6章</td>
    <td>逻辑斯谛回归与最大熵模型77-94页</td>
    <td>10.30-11.05</td>
    <td> </td>
    </tr> 
     <tr>
    <td>第7章</td>
    <td>支持向量机95-134页</td>
    <td>11.06-11.19</td>
    <td>  </td>
    </tr>
    <tr>
    <td>第8章</td>
    <td>提升方法137-153页</td>
    <td>11.20-11.26</td>
    <td>  </td>
    </tr>
        <tr>
    <td>第9章</td>
    <td>EM算法及其推广155-170页</td>
    <td>11.27-12.03</td>
    <td>  </td>
    </tr>
     <tr>
    <td>第10章</td>
    <td>隐马尔可夫模型171-189页</td>
    <td>12.04-12.17</td>
    <td> </td>
    </tr> 
     <tr>
    <td>第11章</td>
    <td>条件随机场191-210页</td>
    <td>12.18-12.31</td>
    <td>  </td>
    </tr>
  </table>
  </div>


<hr style=" height:2px;border:none;border-top:2px dotted #185598;" />

<h2>Assignment 101 </h2>
<div>
<p>
<b>练习内容：</b>熟练 KNN（K 邻近）算法的思想和编码实现
</p>

<p>
<b>详细说明：</b>利用 python 语言实现 KNN 算法，并对收集的鸢尾花数据进行分类。鸢尾花数据
分为两部分，一部分为 104 个带类别标记的样本组成的训练集，另一部分为不带类别标记的
46 个样本组成的测试集，要求对 46 个不带类别标记的样本进行预测。文件夹内包内含4个文件，
分别为 data-description.txt, training.txt, test.txt,knn.py。knn.py 为需要你来完成的代码。
</p>
<p>
<b>输入与输出：</b>输入训练集与测试集数据（见文本文件），输出分类正确率，召回率，F1
</p>
<P>
<b>注意：</b>请尽量不要调用第三方的直接实现，如scikit-learn
</P>
<P>
<b>Deadline：</b>2017-10-31
</P>
</div>

<hr>

<h2>Assignment 102 </h2>
<div>
<p>
<b>练习内容：</b>熟练回归模型的思想和编码实现，掌握梯度下降和正则化方法。
</p>

<p>
<b>详细说明：</b>使用python语言实现回归模型，利用收集的NACA0012 airfoils数据对模型进行训练。
Airfoils 数据分为两部分，一部分为1052 个包含sound pressure level连续值输出的样本组成的训练集，
另一部分为不包含sound pressure level输出的451个样本组成的测试集，要求对451个不带输出值的样本进行输出预测。
文件夹内包含四个文件，分别为data-description.txt, training.txt, test.txt, linear.py。
其中linear.py为需要你来完成的代码。
</p>
<p>
<b>输入与输出：</b>输入训练集（6维）与测试集（5维）数据（见文本文件），输出为预测值与真实值的误差平方和。误差越小，则拟合越好。
</p>
<p>
<b>提示:</b>可以分别尝试用线性模型和多项式模型拟合训练数据，并用梯度下降法来求解模型参数，找到表现更好的模型。如果选择高阶多项式模型，请注意使用正则化项来避免模型过拟合，以提升模型在测试集上的表现。
</p>
<P>
<b>注意：</b>请尽量不要调用第三方的直接实现，如scikit-learn
</P>
<P>
<b>Deadline：</b>2017-11-20
</P>
</div>
