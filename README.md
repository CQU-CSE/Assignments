# Assignments
Assignments for newbies (students of Prof. Gao)

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
<b>提示:</b>提示：可以分别尝试用线性模型和多项式模型拟合训练数据，并用梯度下降法来求解模型参数，找到表现更好的模型。如果选择高阶多项式模型，请注意使用正则化项来避免模型过拟合，以提升模型在测试集上的表现。
</p>
<P>
<b>注意：</b>请尽量不要调用第三方的直接实现，如scikit-learn
</P>
<P>
<b>Deadline：</b>2017-11-20
</P>
</div>
