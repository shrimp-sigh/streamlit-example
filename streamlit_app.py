import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

from PIL import Image

# 自定义 CSS 样式
custom_css = """
<style>
/* 主体样式 */
body {
    background: linear-gradient(to right, #405de6, #5851db, #833ab4, #c13584, #e1306c, #fd1d1d, #f56040, #f77737, #fcaf45, #ffdc80);
    color: #7fffd4;
    font-family: 'Arial';
}

/* 标题样式 */
h1 {
    font-size: 3.5em;
    color: #7fffd4; /* 浅蓝色 */
    text-shadow: 2px 2px 4px #000000;
    text-align: center; /* 让标题居中 */
}

/* 段落样式 */
p {
    font-size: 1.2em;
    color: #000000;
    font-weight: bold;
}


/* 按钮样式 */
.stButton>button {
    color: #000000;
    border-radius: 30px;
    border: none;
    font-weight: bold;
    padding: 15px 30px;
    text-shadow: none;
    font-size: 30px;
}
.stButton>button:hover {
    background-color: #5851db;
    
}

/* 文本输入框样式 */
.stTextInput>div>div>input {
    background-color: rgba(255, 255, 255, 0.7);
    background-color: rgba(0, 0, 0, 1);
    border: none;
    border-radius: 20px;
    padding: 10px 15px;
    box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.stTextInput>div>div>input:focus {
    background-color: rgba(0, 0, 0, 1);
    box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
}
</style>
"""

# 在页面中注入自定义 CSS 样式
st.markdown(custom_css, unsafe_allow_html=True)

data = pd.read_csv("task_metrics.csv")

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(
    data[['watch_time_sum', 'unique_watchers', 'completion_rate', 'total_watch_count', 'repetition_rate']])
data_normalized = pd.DataFrame(data_normalized,
                               columns=['watch_time_sum', 'unique_watchers', 'completion_rate', 'total_watch_count',
                                        'repetition_rate'])

quality_score = (data_normalized['watch_time_sum'] * 10 +
                 data_normalized['unique_watchers'] * 5 +
                 data_normalized['total_watch_count'] * 5 +
                 data_normalized['repetition_rate'] * 0.01 +
                 data_normalized['completion_rate'] * 0.2)

X = data_normalized[['watch_time_sum', 'unique_watchers', 'completion_rate', 'total_watch_count', 'repetition_rate']]
y = quality_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model2 = DecisionTreeRegressor(random_state=42)

# 训练模型
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
target_X = model2.predict(X_test)
target_X2 = model1.predict(X_test)
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
    st.balloons()
    st.snow()
    st.title('教育大数据分析系统')
    st.header('1. 背景介绍')

    st.markdown('''
        <p style="font-size: 1.5em; color: #000000;text-shadow: none;">
        在数字化时代背景下，在线教育已成为教育领域的新趋势。
        大量在线教育平台积累了丰富的学习数据，这些数据蕴含着提高教学质量和学习效果的关键信息。
        为了充分利用这些数据资源，本项目旨在开发一个基于Web的在线教育综合大数据分析系统，
        通过对学习对象、学习内容和学习质量等数据的深入分析，为在线教育平台提供数据驱动的决策支持，
        助力平台优化教学内容，提升用户体验。
        </p>
    ''', unsafe_allow_html=True)

    st.header('2. 功能介绍')

    st.markdown('''
            <p style="font-size: 1.5em; color: #000000;text-shadow: none;">
        系统的主要功能包括数据收集与预处理、数据可视化展示、课程质量评分模型构建。
        首先，系统将收集在线教育平台的用户学习数据，
        包括课程评分、完播率、观看人数、播放量和完成率等关键指标。其次，通过数据可视化技术
        ，系统将直观展示这些指标，帮助教育机构快速把握课程的整体表现。最后，系统将基于总观看时间、观看人数、播放量和完成率这四个因素，
        构建线性回归模型和决策树模型，为课程质量提供量化评估，从而为讲师提供针对性的教学反馈，优化教学方法，提升教学效果。
            </p>
        ''', unsafe_allow_html=True)
    st.header('3. 数据集来源')

    image = Image.open("gyy.png")

    st.image(image,
             caption='标题',
             width=800
             )

    st.markdown('''
                <p style="font-size: 1.5em; color: #000000;text-shadow: none;">
        数据集来源于泰迪云课堂平台，这些数据包括了用户的学习行为、课程互动情况等多个维度。
        通过对这些数据进行深入分析，我们可以揭示课程质量与用户行为之间的关系，
        为在线教育平台提供精准的数据支持和决策依据。此次分析将紧密围绕该平台的实际需求展开，
        确保分析结果能够有效指导平台的教学改进和运营策略制定。通过这样的分析，
        我们可以为在线教育平台提供辅助决策支持，帮助它们更好地满足用户需求，提升教育服务质量。
                </p>
            ''', unsafe_allow_html=True)
    st.markdown('''
                    <p style="font-size: 1.5em; color: #ff0000;">
    下面提供课程评分标准的饼图，通过给总观看时间、观看人数、播放量和完成率这四个因素不同权重，令评估结果更符合实际
                    </p>
                ''', unsafe_allow_html=True)

# 页面1

def page1():
    st.title('课程评分标准')
    labels = ['总观看时间', '观看人数', '播放量', '完播率', '重播率']
    values = [550, 200, 210, 1, 20]
    trace = go.Pie(labels=labels, values=values, hole=0.5)
    table0 = go.Figure(data=trace)
    st.plotly_chart(table0, use_container_width=True)
    st.write(
        "用户调查结果显示：多数人认为一个课程的好坏主要取决于这一课程的总观看时间长短，其次就是播放量与观看人数，因此我们在评估课程质量时提高这三个数据在课程评分中的权重，使得课程评估结果更合理")
    st.markdown('''
                        <p style="font-size: 1.5em; color: #ff0000;text-shadow: none;">
        下面提供四个表格，分别直观展示展示总观看时间、观看人数、播放量和完成率的情况
                        </p>
                    ''', unsafe_allow_html=True)
def page2():
    st.title('表格数据')
    # 使用 Plotly 创建线型图
    # tb1 = px.line(data,y='completion_rate', title='播放率数据', line_shape='linear')
    # tb1.update_xaxes(title_text="课程号")
    # tb1.update_yaxes(title_text="完播率")
    # tb1.update_layout(legend_title_text='quality_score')#添加图例
    # st.plotly_chart(tb1) #在页面中显示图表

    # table1 = go.Figure()
    # x1 = data['completion_rate']
    # table1.add_trace(go.Histogram(x=x1,name='完播率'))
    # table1.update_layout(barmode='stack',title='完播率分布')
    # table1.update_traces(opacity=0.8)
    # table1.update_xaxes(title_text="完播率")
    # table1.update_yaxes(title_text="课程数量")
    # st.plotly_chart(table1)#

    image = Image.open("1.png")

    st.image(image,
             caption='完播率分析',
             width=800
             )

    st.markdown('''
        <div style="text-align: center;">
            <p style="font-size: 1.5em; color: #000000; text-shadow: none;font-weight: bold;">
            (1)根据表格中的数据，completion_rate列的数值范围从0到1，平均值约为0.652。
            通过直方图显示completion_rate的分布情况。我们可以看到有两个主要的峰值，
            一个在0附近，表示有一部分课程几乎没有完成者；另一个峰值接近1，表示有相当一部分课程被完全完成。
            而箱形图则展示了数据的分布范围和中位数。从箱形图中可以观察到，
            有一些课程的完成率为0，而中位数在0.8左右，表明大多数课程至少完成了80%。
            </p>
        </div>
    ''', unsafe_allow_html=True)

    # table2 = go.Figure()
    # x2 = data['unique_watchers']
    # x3 = data['total_watch_count']
    # x4 = data['repetition_rate']
    # table2.add_trace(go.Histogram(x=x2, name='课程观看人数分布'))
    # table2.update_layout(barmode='stack', title='观看人数分布')
    # table2.update_traces(opacity=0.8)
    # table2.update_xaxes(title_text="课程观看人数")
    # table2.update_yaxes(title_text="课程数量")
    # st.plotly_chart(table2)
    image = Image.open("2.png")

    st.image(image,
             caption='观看人数分析',
             width=800
             )

    st.markdown('''
            <div style="text-align: center;">
                <p style="font-size: 1.5em; color: #000000; text-shadow: none;font-weight: bold;">
                    (2)接下来，我们将对unique_watchers这一列进行分析。
                    根据unique_watchers列的基本统计信息，平均独立观看者数量为6.85，标准差较大，为24.1，
                    表明数据分布具有较大的变异性。
                    观看者人数从1人到465人不等，中位数仅为2人，这表明大多数课程的观看者数量相对较少，
                    但存在少数几个课程有非常多的观看者。
                    接下来，我们将通过绘制直方图和箱形图来更详细地查看这个分布情况。
                    这些图表将帮助我们可视化不同课程中独立观看者数量的频率和范围。
                    让我们来看看这些图表。 
                    在这两个图表中，我们可以看到以下信息：
                    •直方图显示unique_watchers的分布是右偏的，大多数课程的观看者数量很少，
                    但有少数课程有异常高的观看者数。
                    这种分布表明，大多数课程的吸引力相对有限，只吸引了少量独立观看者。
                    •箱形图进一步揭示了这种分布的特点，其中50%的数据都集中在5个以下的观看者数，
                    但同时存在一些极端的离群值，这表示有些课程非常受欢迎。
                </p>
            </div>
        ''', unsafe_allow_html=True)

# 页面2
def page3():
    st.title('表格数据')
    # table3 = go.Figure()
    # x3 = data['total_watch_count']
    # x4 = data['repetition_rate']
    # table3.add_trace(go.Histogram(x=x3, name='课程播放量分布'))
    # table3.update_layout(barmode='stack', title='播放量分布')
    # table3.update_traces(opacity=0.8)
    # table3.update_xaxes(title_text="课程播放量")
    # table3.update_yaxes(title_text="课程数量")
    # st.plotly_chart(table3)
    image = Image.open("4.png")
    st.image(image,
             caption='播放量分析',
             width=800
             )
    table4 = go.Figure()

    st.markdown('''
                <div style="text-align: center;">
                    <p style="font-size: 1.5em; color: #000000; text-shadow: none;font-weight: bold;">
                       (3)这是 "total_watch_count" 列的基本统计数据和分布图：在总共744个课程中，其平均观看次数为1335.74次，
                       观看次数的标准差约为4684.43，显示出数据的巨大波动。其中最小的观看值仅为1次，其中有25%的课程观看次数不超过31次，
                       有一半的课程观看次数低于310次，
                       有75%的课程观看次数不超过1059次，但是其中最大的观看次数为92211次，从分布图中可以看出，
                       绝大多数课程的观看次数都较低，但存在一些极端的高值。这些极端值表明仅有少数课程极受欢迎。
                    </p>
                </div>
            ''', unsafe_allow_html=True)
    # x4 = data['repetition_rate']
    # table4.add_trace(go.Histogram(x=x4, name='课程重播率分布'))
    # table4.update_layout(barmode='stack', title='重播率分布')
    # table4.update_traces(opacity=0.8)
    # table4.update_xaxes(title_text="课程重播率")
    # table4.update_yaxes(title_text="课程数量")
    # st.plotly_chart(table4)
    image = Image.open("3.png")
    st.image(image,
             caption='播放量分析',
             width=800
             )

    st.markdown('''
                   <div style="text-align: center;">
                       <p style="font-size: 1.5em; color: #000000; text-shadow: none;font-weight: bold;">
                        (4)基本统计信息显示，这个列的均值为201.36，标准差为321.18，这表明观看行为的重复频率具有很大的差异。
                        最小值为1，而最大值达到了3781，这表明某些课程可能非常受欢迎或者内容引人回看。
                        接下来，我们使用直方图和箱形图来进一步探索这种分布的特性。
                        这些图表将帮助我们更好地理解数据的总体趋势和异常值。
                        •直方图揭示了重复观看率的分布是极度右偏的，多数课程的重复观看率较低，
                        但也有一些极端的高值，表明少数课程被反复观看的次数非常高。
                        •箱形图进一步证实了这一点，显示大多数数据点集中在较低的重复观看率区间，
                        但存在大量离群值，这意味着有些课程的内容可能极具吸引力，促使观看者多次回看。
                       </p>
                   </div>
               ''', unsafe_allow_html=True)


def page4():
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
        user_input_normalized = scaler.transform(
            [[watch_time_sum, unique_watchers, completion_rate, total_watch_count, repetition_rate]])
        quality_score_pred = model.predict(user_input_normalized)

        all_scores_with_pred = np.append(y, quality_score_pred)
        rank_of_pred = np.argsort(np.argsort(-all_scores_with_pred))[-1] + 1

        st.write(f'课程质量得分: {quality_score_pred[0]}，在所有744个课程质量得分中的排名为: {rank_of_pred}')


# 检测session_state中是否有page这个键，没有则初始化page键为“home”
with st.sidebar:  # 在侧边栏中创建菜单以导航到不同的页面
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 3.5em;'>导航栏</h1>", unsafe_allow_html=True)
    if st.button("系统概览"):
        st.session_state.page = "home"  # 将值赋给session_state.page
    if st.button("课程评分标准"):
        st.session_state.page = "page1"
    if st.button("表格一二"):
        st.session_state.page = "page2"
    if st.button("表格三四"):
        st.session_state.page = "page3"
    if st.button("课程质量评估"):
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
