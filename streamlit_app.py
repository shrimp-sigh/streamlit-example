import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

data = pd.read_csv("task_metrics.csv")


scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data[['watch_time_sum', 'unique_watchers',  'completion_rate','total_watch_count','repetition_rate']])
data_normalized = pd.DataFrame(data_normalized, columns=['watch_time_sum', 'unique_watchers', 'completion_rate','total_watch_count','repetition_rate'])
print(data_normalized)
quality_score = (data_normalized['watch_time_sum']*10+
                 data_normalized['unique_watchers']*5+
                 data_normalized['total_watch_count']*5+
                 data_normalized['repetition_rate']*0.01+
                 data_normalized['completion_rate']*0.2)

X = data_normalized[['watch_time_sum', 'unique_watchers', 'completion_rate','total_watch_count','repetition_rate']]
y = quality_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)






# 使用Streamlit创建Web应用程序

if 'page' not in st.session_state:
    st.session_state.page = "home"
#首页
def home():
    st.balloons()
    st.snow()
    st.title('课程质量评估')
    # 用户输入
    watch_time_sum = st.number_input('总观看时间')
    unique_watchers = st.number_input('观看人数')
    total_watch_count = st.number_input('播放量')
    completion_rate = st.number_input('完成率')

    # 添加按钮
    btn_generate_score = st.button('生成课程质量评分')
    selected_model = st.radio("选择模型", ("线性回归模型", "决策树模型"))

    if selected_model == "线性回归模型":
        model = LinearRegression()
    elif selected_model == "决策树模型":
        model = DecisionTreeRegressor(random_state=42)

    model.fit(X_train, y_train)

    if btn_generate_score:
        if btn_generate_score:
            repetition_rate = unique_watchers * completion_rate  # 这里可以根据需要进行具体的计算

            user_input_normalized = scaler.transform(
                [[watch_time_sum, unique_watchers, completion_rate, total_watch_count,repetition_rate]])

            quality_score_pred = model.predict(user_input_normalized)

            all_scores_with_pred = np.append(y, quality_score_pred)
            rank_of_pred = np.argsort(np.argsort(-all_scores_with_pred))[-1] + 1

            st.write(f'课程质量得分: {quality_score_pred[0]}，在所有744个课程质量得分中的排名为: {rank_of_pred}')

#页面1
def page1():
    st.write("表格数据")
    # 使用 Plotly 创建线型图
    tb1 = px.line(data,y='completion_rate', title='播放率数据', line_shape='linear')
    tb1.update_xaxes(title_text="课程号")
    tb1.update_yaxes(title_text="完播率")
    tb1.update_layout(legend_title_text='quality_score')#添加图例
    st.plotly_chart(tb1) #在页面中显示图表

    tb2 = px.line(data, y='unique_watchers',title='用户观看人数',line_shape='linear')
    tb2.update_xaxes(title_text="课程号")
    tb2.update_yaxes(title_text="有几个用户观看了此视频")
    tb2.update_layout(legend_title_text='quality_score')
    st.plotly_chart(tb2)

def page2():
    st.write("表格数据")
    tb1 = px.line(data, y='total_watch_count', title='视频播放量数据', line_shape='linear')
    tb1.update_xaxes(title_text="课程号")
    tb1.update_yaxes(title_text="播放量")
    tb1.update_layout(legend_title_text='quality_score')  # 添加图例
    st.plotly_chart(tb1)  # 在页面中显示图表

    tb2 = px.line(data, y='repetition_rate', title='视频重播率数据', line_shape='linear')
    tb2.update_xaxes(title_text="课程号")
    tb2.update_yaxes(title_text="重播率")
    tb2.update_layout(legend_title_text='quality_score')
    st.plotly_chart(tb2)
#检测session_state中是否有page这个键，没有则初始化page键为“home”



with st.sidebar: #在侧边栏中创建菜单以导航到不同的页面
    st.header("导航栏") #侧栏标题
    if st.button("首页"):
        st.session_state.page = "home" #将值赋给session_state.page
    if st.button("第二页"):
        st.session_state.page = "page1"
    if st.button("第三页"):
        st.session_state.page = "page2"

#这一步实现页面跳转
if st.session_state.page == "home":
    home()
elif st.session_state.page == "page1":
    page1()
elif st.session_state.page == "page2":
    page2()
