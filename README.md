# bankmarketing_classification
Machine learning
## 项目背景
### 数据来源 
数据选自UCI机器学习库中的「银行营销数据集(Bank Marketing Data Set)。
这些数据与葡萄牙银行机构的营销活动相关。这些营销活动以电话为基础，一般，银行的客服人员需要联系客户至少一次，以此确认客户是否将认购该银行的产品（定期存款）。
### 数据字段说明

NO | 字段名称 | 数据类型 | 字段描述
-|-|-|-
1 | ID | Int | 客户唯一标识 
2	| age |	Int | 客户年龄
3 |	job|	String|	客户的职业
4 |	marital |	String|	婚姻状况
5 |	education |	String |	受教育水平
6	| default |	String | 是否有违约记录
7	| balance |	Int	 | 每年账户的平均余额
8	| housing |	String	 | 是否有住房贷款
9	| loan |	String |	是否有个人贷款
10	| contact |	String |	与客户联系的沟通方式
11	| day	| Int	 | 最后一次联系的时间（几号）
12	| month	| String | 	最后一次联系的时间（月份）
13	| duration	| Int	 | 最后一次联系的交流时长
14	| campaign	| Int	| 在本次活动中，与该客户交流过的次数
15	| pdays |	Int	 | 距离上次活动最后一次联系该客户，过去了多久（999表示没有联系过）
16	| previous |	Int	 |在本次活动之前，与该客户交流过的次数
17	| poutcome |	String | 上一次活动的结果
18	| y |	Int	| 预测客户是否会订购定期存款业务

### 问题描述
  与该数据集对应的任务是「分类任务」，「分类目标」是预测客户是(' 1 ')或者否(' 0 ')购买该银行的产品。
### 方法
逻辑分类
## 安装
* 环境：python3
* IDE:VS code
* 相关库：
  * pandas——数据清洗
  * numpy——构造特征
  * sklearn——数据挖掘 数据分析
  ## 使用
  运行data.py
  ## 声明
  如有问题，请及时联系：fennyh@csu.edu.cn
