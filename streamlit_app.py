import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st
import plotly.express as px
data = pd.read_csv("D:\\PCdesk\\competition\\image\\task_metrics.csv")


scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data[['watch_time_sum', 'unique_watchers',  'completion_rate','total_watch_count','repetition_rate']])
data_normalized = pd.DataFrame(data_normalized, columns=['watch_time_sum', 'unique_watchers', 'completion_rate','total_watch_count','repetition_rate'])

quality_score = (data_normalized['watch_time_sum']*10+
                 data_normalized['unique_watchers']*5+
                 data_normalized['total_watch_count']*5+
                 data_normalized['repetition_rate']*0.01+
                 data_normalized['completion_rate']*0.2)

X = data_normalized[['watch_time_sum', 'unique_watchers', 'completion_rate','total_watch_count','repetition_rate']]
y = quality_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model1 = LinearRegression()
model2 = DecisionTreeRegressor(random_state=42)

# 训练模型
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
target_X=model2.predict(X_test)
target_X2=model1.predict(X_test)
print(target_X[:10])
print("XXXXXXXXXXXXX")
print(target_X2[:10])
print("XXXXXXXXXXXXX")
print(y_test.head(10))

# 使用Streamlit创建Web应用程序
if 'page' not in st.session_state:
    st.session_state.page = "home"

# 首页
def home():
    st.title('课程质量评估')
    # 用户输入
    watch_time_sum = st.number_input('总观看时间')
    unique_watchers = st.number_input('观看人数')
    total_watch_count = st.number_input('播放量')
    completion_rate = st.number_input('完成率')

    # 添加按钮
    btn_generate_score = st.button('生成课程质量评分')
    selected_model = st.radio("选择模型", ("线性回归模型", "决策树模型"))

    if btn_generate_score:
        if selected_model == "线性回归模型":
            model = model1
        elif selected_model == "决策树模型":
            model = model2

        repetition_rate = unique_watchers * completion_rate  # 计算 repetition_rate
        user_input_normalized = scaler.transform([[watch_time_sum, unique_watchers, completion_rate, total_watch_count,repetition_rate]])
        quality_score_pred = model.predict(user_input_normalized)

        all_scores_with_pred = np.append(y, quality_score_pred)
        rank_of_pred = np.argsort(np.argsort(-all_scores_with_pred))[-1] + 1

        st.write(f'课程质量得分: {quality_score_pred[0]}，在所有744个课程质量得分中的排名为: {rank_of_pred}')

# 页面1

def page1():
    st.write("课程评分标准")
    labels = ['总观看时间','观看人数','播放量','完播率','重播率']
    values = [550,200,210,1,20]
    trace = go.Pie(labels=labels, values=values,hole=0.5)
    table0 = go.Figure(data=trace)
    st.plotly_chart(table0, use_container_width=True)
    st.write("用户调查结果显示：多数人认为一个课程的好坏主要取决于这一课程的总观看时间长短，其次就是播放量与观看人数，因此我们在评估课程质量时提高这三个数据在课程评分中的权重，使得课程评估结果更合理")
def page2():
    st.write("表格数据")
    # 使用 Plotly 创建线型图
    #tb1 = px.line(data,y='completion_rate', title='播放率数据', line_shape='linear')
    #tb1.update_xaxes(title_text="课程号")
    #tb1.update_yaxes(title_text="完播率")
    #tb1.update_layout(legend_title_text='quality_score')#添加图例
    #st.plotly_chart(tb1) #在页面中显示图表

    table1 = go.Figure()
    x1 = data['completion_rate']
    table1.add_trace(go.Histogram(x=x1,name='完播率'))
    table1.update_layout(barmode='stack',title='完播率分布')
    table1.update_traces(opacity=0.8)
    table1.update_xaxes(title_text="完播率")
    table1.update_yaxes(title_text="课程数量")
    st.plotly_chart(table1)

    table2 = go.Figure()
    x2 = data['unique_watchers']
    x3 = data['total_watch_count']
    x4 = data['repetition_rate']
    table2.add_trace(go.Histogram(x=x2, name='课程观看人数分布'))
    table2.update_layout(barmode='stack', title='观看人数分布')
    table2.update_traces(opacity=0.8)
    table2.update_xaxes(title_text="课程观看人数")
    table2.update_yaxes(title_text="课程数量")
    st.plotly_chart(table2)

# 页面2
def page3():
    table3 = go.Figure()
    x3 = data['total_watch_count']
    x4 = data['repetition_rate']
    table3.add_trace(go.Histogram(x=x3, name='课程播放量分布'))
    table3.update_layout(barmode='stack', title='播放量分布')
    table3.update_traces(opacity=0.8)
    table3.update_xaxes(title_text="课程播放量")
    table3.update_yaxes(title_text="课程数量")
    st.plotly_chart(table3)

    table4 = go.Figure()
    x4 = data['repetition_rate']
    table4.add_trace(go.Histogram(x=x4, name='课程重播率分布'))
    table4.update_layout(barmode='stack', title='重播率分布')
    table4.update_traces(opacity=0.8)
    table4.update_xaxes(title_text="课程重播率")
    table4.update_yaxes(title_text="课程数量")
    st.plotly_chart(table4)
def page4():
    # 读取数据
    data = pd.read_csv("D:\\PCdesk\\competition\\data\\log.csv")

    # 计算每个类别的计数信息
    category_distribution = data['action'].value_counts()

    # 取出现次数最多的4个类型
    top_4_categories = category_distribution.head(4)

    # 计算其他类型的总计数
    other_count = category_distribution.sum() - top_4_categories.sum()

    # 将其他类型的计数合并到“其他”类型中
    top_4_categories['其他'] = other_count

    # 打印分类分布情况
    st.write(top_4_categories)

    # 使用 Plotly 绘制饼图
    fig = px.pie(values=top_4_categories, names=top_4_categories.index, title='日志类型')
    st.plotly_chart(fig)
    st.write("由日志表操作类型可见，大多数用户都是可以登录成功的，可见该平台系统运行稳定")

# 检测session_state中是否有page这个键，没有则初始化page键为“home”
with st.sidebar: #在侧边栏中创建菜单以导航到不同的页面
    st.header("导航栏") #侧栏标题
    if st.button("首页"):
        st.session_state.page = "home" #将值赋给session_state.page
    if st.button("第二页"):
        st.session_state.page = "page1"
    if st.button("第三页"):
        st.session_state.page = "page2"
    if st.button("第四页"):
        st.session_state.page = "page3"
    if st.button("日志表分析"):
        st.session_state.page = "page4"


# 这一步实现页面跳转
if st.session_state.page == "home":
    home()
elif st.session_state.page == "page1":
    page1()
elif st.session_state.page == "page2":
    page2()
elif st.session_state.page == "page3":
    page3()
elif st.session_state.page == "page4":
    page4()
