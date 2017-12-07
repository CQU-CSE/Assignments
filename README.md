# Assignments
Assignments for newbies (students of Prof. Gao)

<h2>Study Plan 学习计划</h2>
<p>以下为针对2017-2018学年研一及保研同学的《统计学习方法》一书的阅读进度安排。</br>
<b>注:</b>《统计学习方法》中有少许错误，可查看<a href="http://blog.csdn.net/wzmsltw/article/details/52718722">勘误表</a>修正。
</p>
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
<hr>

<h2>Assignment 103 </h2>
<div>
<p>
<b>练习内容：</b>音乐播放量预测，尝试多模型结合解决问题
</p>

<p align="left">
	经过7年的发展与沉淀，目前阿里音乐拥有数百万的曲库资源，每天千万的用户活跃在平台上，拥有数亿人次的用户试听、收藏等行为。在原创艺人和作品方面，更是拥有数万的独立音乐人，每月上传上万个原创作品，形成超过几十万首曲目的原创作品库，如此庞大的数据资源库对于音乐流行趋势的把握有着极为重要的指引作用。</p>
<p>
	本次练习以阿里音乐用户的历史播放数据为基础，期望同学们可以通过对阿里音乐平台上每个阶段艺人的试听量的预测，挖掘出即将成为潮流的艺人，从而实现对一个时间段内音乐流行趋势的准确把控。</p>
<b>数据说明</b></br>
<p>
	数据集包含抽样的歌曲艺人数据，以及和这些艺人相关的6个月内（20150301-20150830）的用户行为历史记录。<a href="https://pan.baidu.com/s/1eRR3fAM">数据集下载</a></p>
<b>用户行为表（mars_tianchi_user_actions）</b></br>
<table>
	<tbody>
		<tr>
			<td style="width: 108px;">
				<p align="left">
					列名</p>
			</td>
			<td style="width: 61px;">
				<p align="left">
					类型</p>
			</td>
			<td style="width: 216px;">
				<p align="left">
					说明</p>
			</td>
			<td style="width: 183px;">
				<p align="left">
					示例</p>
			</td>
		</tr>
		<tr>
			<td style="width: 108px;">
				<p align="left">
					user_id</p>
			</td>
			<td style="width: 61px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 216px;">
				<p align="left">
					用户唯一标识</p>
			</td>
			<td style="width: 183px;">
				<p align="left">
					7063b3d0c075a4d276c5f06f4327cf4a</p>
			</td>
		</tr>
		<tr>
			<td style="width: 108px;">
				<p align="left">
					song_id</p>
			</td>
			<td style="width: 61px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 216px;">
				<p align="left">
					歌曲唯一标识</p>
			</td>
			<td style="width: 183px;">
				<p align="left">
					effb071415be51f11e845884e67c0f8c</p>
			</td>
		</tr>
		<tr>
			<td style="width: 108px;">
				<p align="left">
					gmt_create</p>
			</td>
			<td style="width: 61px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 216px;">
				<p align="left">
					用户播放时间（unix时间戳表示）精确到小时</p>
			</td>
			<td style="width: 183px;">
				<p align="left">
					1426406400</p>
			</td>
		</tr>
		<tr>
			<td style="width: 108px;">
				<p align="left">
					action_type</p>
			</td>
			<td style="width: 61px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 216px;">
				<p align="left">
					行为类型：1，播放；2，下载，3，收藏</p>
			</td>
			<td style="width: 183px;">
				<p align="left">
					1</p>
			</td>
		</tr>
		<tr>
			<td style="width: 108px;">
				<p align="left">
					Ds</p>
			</td>
			<td style="width: 61px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 216px;">
				<p align="left">
					记录收集日（分区）</p>
			</td>
			<td style="width: 183px;">
				<p align="left">
					20150315</p>
			</td>
		</tr>
	</tbody>
</table>
<p align="left">
	注：用户对歌曲的任意行为为一行数据。</p>
<b>歌曲艺人（mars_tianchi_songs）</b></br>
<table>
	<tbody>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					列名</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					类型</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					说明</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					示例</p>
			</td>
		</tr>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					song_id</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					歌曲唯一标识</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					c81f89cf7edd24930641afa2e411b09c</p>
			</td>
		</tr>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					artist_id</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					歌曲所属的艺人Id</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					03c6699ea836decbc5c8fc2dbae7bd3b</p>
			</td>
		</tr>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					publish_time</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					歌曲发行时间，精确到天</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					20150325</p>
			</td>
		</tr>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					song_init_plays</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					歌曲的初始播放数，表明该歌曲的初始热度</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					0</p>
			</td>
		</tr>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					Language</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					数字表示1,2,3…</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					100</p>
			</td>
		</tr>
		<tr>
			<td style="width: 130px;">
				<p align="left">
					Gender</p>
			</td>
			<td style="width: 58px;">
				<p align="left">
					String</p>
			</td>
			<td style="width: 206px;">
				<p align="left">
					1,2,3</p>
			</td>
			<td style="width: 173px;">
				<p align="left">
					1</p>
			</td>
		</tr>
	</tbody>
</table>
<b>任务</b></br>
<p align="left">
	同学们需要自行划分训练集测试集，预测艺人随后2个月，即60天（20150701-20150830）的播放数据。注：训练集内不能包含后两个月的用户行为数据</p>
<b>评价指标</b></br>
<p><img src="https://img.alicdn.com/tps/TB12LRiMXXXXXbKaXXXXXXXXXXX-801-341.png"></p>
<p>
<b>Deadline：2018-01-20</b>
</p>
</div>
