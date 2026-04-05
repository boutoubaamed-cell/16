import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, f_oneway, ttest_ind, mannwhitneyu, kruskal, pearsonr, spearmanr, chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pingouin as pg

# إعداد الخط العربي للمخططات
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="نظام التحليل الإحصائي المتقدم للاستبيانات",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق الصفحة
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .analysis-box {
        background-color: #e8f4f8;
        border-right: 4px solid #2e86ab;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .trend-strongly-agree {
        background-color: #1b5e20;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .trend-agree {
        background-color: #4caf50;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .trend-neutral {
        background-color: #ffc107;
        color: #333;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .trend-disagree {
        background-color: #ff9800;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .trend-strongly-disagree {
        background-color: #f44336;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .factor-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-right: 5px solid #2e86ab;
    }
</style>
""", unsafe_allow_html=True)


# دالة لتحديد الاتجاه العام حسب سلم ليكارت الخماسي
def get_likert_trend(mean_value):
    """تحديد الاتجاه حسب سلم ليكارت الخماسي"""
    if mean_value >= 4.5:
        return "موافق بشدة ✅✅", "trend-strongly-agree", "مرتفع جداً"
    elif mean_value >= 3.5:
        return "موافق ✅", "trend-agree", "مرتفع"
    elif mean_value >= 2.5:
        return "محايد ➖", "trend-neutral", "متوسط"
    elif mean_value >= 1.5:
        return "غير موافق ❌", "trend-disagree", "منخفض"
    else:
        return "غير موافق بشدة ❌❌", "trend-strongly-disagree", "منخفض جداً"


# دالة لتوليد ملخص للمحور
def generate_factor_summary(factor_name, mean_score, trend_text, trend_level, cronbach_alpha, questions_list):
    """توليد ملوص للمحور بناءً على النتائج"""

    if trend_level == "مرتفع جداً":
        interpretation = f"يتمتع أفراد العينة بمستوى {trend_level} من {factor_name}، حيث بلغ المتوسط العام {mean_score:.2f} من 5."
    elif trend_level == "مرتفع":
        interpretation = f"يظهر أفراد العينة مستوى {trend_level} من {factor_name}، حيث بلغ المتوسط العام {mean_score:.2f} من 5."
    elif trend_level == "متوسط":
        interpretation = f"يتسم أفراد العينة بمستوى {trend_level} من {factor_name}، حيث بلغ المتوسط العام {mean_score:.2f} من 5."
    elif trend_level == "منخفض":
        interpretation = f"يعاني أفراد العينة من مستوى {trend_level} من {factor_name}، حيث بلغ المتوسط العام {mean_score:.2f} من 5."
    else:
        interpretation = f"يعاني أفراد العينة من مستوى {trend_level} جداً من {factor_name}، حيث بلغ المتوسط العام {mean_score:.2f} من 5."

    # تقييم الثبات
    if cronbach_alpha >= 0.8:
        reliability_text = "ممتاز، مما يعزز ثقة النتائج."
    elif cronbach_alpha >= 0.7:
        reliability_text = "مقبول، ويمكن الاعتماد على النتائج."
    elif cronbach_alpha >= 0.6:
        reliability_text = "ضعيف، يوصى بمراجعة بعض الفقرات."
    else:
        reliability_text = "غير مقبول، يجب إعادة النظر في فقرات هذا المحور."

    return f"""
    **المحور: {factor_name}**\n
    📊 **الاتجاه العام:** {trend_text}\n
    📈 **المتوسط العام:** {mean_score:.2f} من 5\n
    🔧 **معامل الثبات (ألفا كرونباخ):** {cronbach_alpha:.3f} - {reliability_text}\n
    📝 **عدد الفقرات:** {len(questions_list)} فقرة\n
    💡 **الملخص:** {interpretation}
    """


# العنوان الرئيسي
st.markdown('<h1 class="main-header">نظام التحليل الإحصائي المتقدم للاستبيانات</h1>', unsafe_allow_html=True)

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ الإعدادات")
    st.markdown("---")

    uploaded_file = st.file_uploader("📁 تحميل ملف Excel", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ تم تحميل البيانات بنجاح! ({df.shape[0]} صف، {df.shape[1]} عمود)")

            st.subheader("📊 معلومات البيانات:")
            st.write(f"**عدد الصفوف:** {df.shape[0]}")
            st.write(f"**عدد الأعمدة:** {df.shape[1]}")
            st.write(f"**القيم المفقودة:** {df.isnull().sum().sum()}")

            with st.expander("📋 عرض أسماء الأعمدة"):
                st.write(list(df.columns))

        except Exception as e:
            st.error(f"❌ خطأ في قراءة الملف: {e}")
            df = None
    else:
        st.info("📌 يرجى تحميل ملف Excel أو CSV يحتوي على بيانات الاستبيان")
        df = None

    st.markdown("---")
    st.markdown("### 📖 تعليمات")
    st.markdown("""
    1. قم بتحميل ملف البيانات
    2. حدد المتغيرات الاجتماعية
    3. حدد المحاور والفقرات
    4. اختر الاختبارات المطلوبة
    5. احصل على النتائج
    """)

if df is not None:

    # الخطوة 1: تحديد المتغيرات
    st.markdown('<div class="section-header">📌 الخطوة 1: تحديد المتغيرات</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="analysis-box">
    <strong>📖 شرح هذه الخطوة:</strong><br>
    في هذه الخطوة، نقوم بتصنيف أعمدة البيانات إلى ثلاثة أنواع:<br>
    • <strong>المتغيرات الاجتماعية والديموغرافية:</strong> مثل العمر، الجنس، المؤهل العلمي، سنوات الخبرة.<br>
    • <strong>المتغيرات المستقلة الإضافية:</strong> متغيرات قد تؤثر على النتائج ولكنها ليست ديموغرافية.<br>
    • <strong>فقرات الاستبيان:</strong> الأسئلة التي تقيس الاتجاهات أو السلوكيات (يتم تحديدها تلقائياً).
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        all_columns = list(df.columns)
        social_vars = st.multiselect(
            "👥 المتغيرات الاجتماعية والديموغرافية:",
            options=all_columns,
            help="اختر الأعمدة التي تمثل المتغيرات الديموغرافية والاجتماعية"
        )

    with col2:
        independent_vars = st.multiselect(
            "📈 المتغيرات المستقلة الإضافية:",
            options=[col for col in all_columns if col not in social_vars],
            help="اختر المتغيرات التي قد تؤثر على النتائج"
        )

    with col3:
        num_factors = st.number_input(
            "🔢 عدد المحاور/المتغيرات الكامنة:",
            min_value=1,
            max_value=15,
            value=2,
            step=1
        )

    if social_vars:
        question_vars = [col for col in all_columns if col not in social_vars + independent_vars]

        st.markdown(f"""
        <div class="success-box">
            <strong>✅ تم التعرف على:</strong><br>
            - المتغيرات الاجتماعية: {len(social_vars)} متغير<br>
            - المتغيرات المستقلة: {len(independent_vars)} متغير<br>
            - فقرات الاستبيان: {len(question_vars)} فقرة
        </div>
        """, unsafe_allow_html=True)

        # الخطوة 2: تحديد المحاور
        st.markdown('<div class="section-header">🎯 الخطوة 2: تحديد المحاور الكامنة</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-box">
        <strong>📖 شرح هذه الخطوة:</strong><br>
        المحاور الكامنة هي أبعاد غير مباشرة تقاس بمجموعة من الفقرات. مثلاً، "الرضا الوظيفي" يمكن قياسه عبر عدة فقرات.<br>
        <strong>كيف تعمل؟</strong> يتم حساب المتوسط الحسابي للفقرات المحددة لكل محور، مما ينتج متغيراً مركباً يمثل ذلك البعد.
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs([f"المحور {i + 1}" for i in range(num_factors)])

        factors = []
        for i, tab in enumerate(tabs):
            with tab:
                col1, col2 = st.columns(2)

                with col1:
                    factor_name = st.text_input(f"اسم المحور {i + 1}:", value=f"المحور_{i + 1}", key=f"name_{i}")
                    selected_questions = st.multiselect(
                        f"📝 الفقرات الممثلة للمحور {i + 1}:",
                        options=question_vars,
                        key=f"questions_{i}",
                        help="اختر الفقرات التي تقيس هذا المحور"
                    )

                with col2:
                    selected_social_vars = st.multiselect(
                        f"👥 المتغيرات الاجتماعية المرتبطة بالمحور {i + 1}:",
                        options=social_vars,
                        key=f"social_{i}",
                        help="اختر المتغيرات الاجتماعية للتحليل مع هذا المحور"
                    )
                    selected_independent = st.multiselect(
                        f"📊 المتغيرات المستقلة المرتبطة بالمحور {i + 1}:",
                        options=independent_vars,
                        key=f"independent_{i}",
                        help="اختر المتغيرات المستقلة للتحليل مع هذا المحور"
                    )

                factors.append({
                    'id': i,
                    'name': factor_name,
                    'questions': selected_questions,
                    'social_vars': selected_social_vars,
                    'independent_vars': selected_independent
                })

        # الخطوة 3: تحديد العلاقات بين المحاور
        st.markdown('<div class="section-header">🔗 الخطوة 3: تحديد العلاقات بين المحاور</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-box">
        <strong>📖 شرح هذه الخطوة:</strong><br>
        هنا نحدد العلاقات السببية بين المحاور لتحليل الانحدار:<br>
        • <strong>المحور التابع (المتغير المعتمد):</strong> هو المتغير الذي نريد تفسيره أو التنبؤ به.<br>
        • <strong>المحاور المستقلة:</strong> هي المتغيرات التي نعتقد أنها تؤثر على المحور التابع.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            factor_names = [f['name'] for f in factors if f['questions']]
            if factor_names:
                dependent_factor = st.selectbox(
                    "🎯 المحور التابع (المتغير المعتمد):",
                    options=factor_names,
                    help="المتغير الذي تريد تفسيره أو التنبؤ به"
                )
            else:
                dependent_factor = None
                st.warning("يرجى تحديد الفقرات للمحاور أولاً")

        with col2:
            independent_factors = st.multiselect(
                "📈 المحاور المستقلة (المتغيرات المستقلة):",
                options=[f for f in factor_names if f != dependent_factor] if dependent_factor else factor_names,
                help="المتغيرات التي تستخدم للتنبؤ بالمحور التابع"
            )

        # الخطوة 4: خيارات التحليل المتقدمة
        st.markdown('<div class="section-header">⚙️ الخطوة 4: خيارات التحليل المتقدمة</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="analysis-box">
        <strong>📖 شرح هذه الخطوة:</strong><br>
        • <strong>مستوى الدلالة:</strong> الاحتمالية المستخدمة لقبول أو رفض الفرضيات (الأكثر شيوعاً 0.05).<br>
        • <strong>الاختبارات اللامعلمية:</strong> تستخدم عندما لا تتبع البيانات التوزيع الطبيعي.<br>
        • <strong>المخططات:</strong> اختيار أنواع الرسوم البيانية التي تريد عرضها.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            significance_level = st.select_slider(
                "📊 مستوى الدلالة الإحصائية:",
                options=[0.01, 0.05, 0.10],
                value=0.05,
                help="المستوى المستخدم لتحديد دلالة النتائج"
            )
            use_nonparametric = st.checkbox(
                "🔄 استخدام الاختبارات اللامعلمية",
                value=False,
                help="يوصى به للبيانات غير الطبيعية أو المقاييس الترتيبية"
            )

        with col2:
            show_correlation = st.checkbox("📈 عرض مصفوفة الارتباط", value=True)
            show_heatmap = st.checkbox("🔥 عرض خريطة الحرارة", value=True)
            show_boxplots = st.checkbox("📦 عرض المخططات الصندوقية", value=True)

        with col3:
            show_regression = st.checkbox("📊 تحليل الانحدار المتقدم", value=True)
            show_efa = st.checkbox("🔍 التحليل العاملي الاستكشافي", value=False)
            show_clustering = st.checkbox("🎯 تحليل التجميع", value=False)

        if st.button("🚀 إجراء التحليل الإحصائي المتقدم", type="primary", use_container_width=True):

            valid_analysis = True
            has_questions = False

            for factor in factors:
                if len(factor['questions']) > 0:
                    has_questions = True

            if not has_questions:
                st.error("❌ يرجى اختيار فقرات لمحور واحد على الأقل")
                valid_analysis = False

            if valid_analysis and has_questions:
                with st.spinner("🔄 جاري إجراء التحليل الإحصائي المتقدم..."):

                    # حساب المتغيرات الكامنة
                    for factor in factors:
                        if factor['questions']:
                            df[factor['name']] = df[factor['questions']].mean(axis=1)

                    # ==================== الجزء 1: الإحصاءات الوصفية ====================
                    st.markdown('<div class="section-header">📊 الجزء 1: الإحصاءات الوصفية للمتغيرات</div>',
                                unsafe_allow_html=True)

                    st.markdown("""
                    <div class="analysis-box">
                    <strong>📈 تحليل النتائج:</strong><br>
                    الإحصاءات الوصفية تلخص البيانات المركزية والتشتت للمتغيرات الرقمية.
                    </div>
                    """, unsafe_allow_html=True)

                    numeric_df = df.select_dtypes(include=[np.number])

                    if len(numeric_df.columns) > 0:
                        desc_stats = numeric_df.describe().T
                        desc_stats['الوسيط'] = numeric_df.median()
                        desc_stats['المدى'] = numeric_df.max() - numeric_df.min()
                        desc_stats['معامل الاختلاف'] = (desc_stats['std'] / desc_stats['mean']) * 100
                        desc_stats['معامل الاختلاف'] = desc_stats['معامل الاختلاف'].replace([np.inf, -np.inf], np.nan)

                        st.dataframe(
                            desc_stats.style.format({
                                'mean': '{:.3f}', 'std': '{:.3f}', 'min': '{:.3f}',
                                '25%': '{:.3f}', '50%': '{:.3f}', '75%': '{:.3f}',
                                'max': '{:.3f}', 'الوسيط': '{:.3f}', 'المدى': '{:.3f}',
                                'معامل الاختلاف': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("📌 لا توجد متغيرات رقمية لعرض الإحصاءات الوصفية.")

                    # ==================== الجزء 2: الاتجاه العام للفقرات (سلم ليكارت) ====================
                    st.markdown(
                        '<div class="section-header">📈 الجزء 2: الاتجاه العام للفقرات (سلم ليكارت الخماسي)</div>',
                        unsafe_allow_html=True)

                    st.markdown("""
                    <div class="analysis-box">
                    <strong>📈 تحليل النتائج:</strong><br>
                    يوضح هذا الجدول الاتجاه العام لكل فقرة من فقرات الاستبيان بناءً على سلم ليكارت الخماسي:<br>
                    • <strong>موافق بشدة (4.5 - 5):</strong> اتجاه إيجابي قوي<br>
                    • <strong>موافق (3.5 - 4.49):</strong> اتجاه إيجابي<br>
                    • <strong>محايد (2.5 - 3.49):</strong> اتجاه متعادل<br>
                    • <strong>غير موافق (1.5 - 2.49):</strong> اتجاه سلبي<br>
                    • <strong>غير موافق بشدة (1 - 1.49):</strong> اتجاه سلبي قوي
                    </div>
                    """, unsafe_allow_html=True)

                    items_trend_data = []
                    for factor in factors:
                        for q in factor['questions']:
                            if q in df.columns:
                                mean_val = df[q].mean()
                                trend_text, trend_class, trend_level = get_likert_trend(mean_val)
                                items_trend_data.append({
                                    'الفقرة': q,
                                    'المحور التابع': factor['name'],
                                    'المتوسط': mean_val,
                                    'الاتجاه العام': trend_text,
                                    'التصنيف': trend_class,
                                    'المستوى': trend_level
                                })

                    if items_trend_data:
                        for idx, row in enumerate(items_trend_data):
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid #eee; background-color: {'#f8f9fa' if idx % 2 == 0 else 'white'}; border-radius: 8px; margin: 2px 0;">
                                <div style="flex: 2;">
                                    <strong>{row['الفقرة']}</strong><br>
                                    <span style="font-size: 0.8rem; color: #666;">{row['المحور التابع']}</span>
                                </div>
                                <div style="flex: 1; text-align: center;">
                                    <span style="font-size: 1.1rem; font-weight: bold;">{row['المتوسط']:.2f}</span>
                                    <span style="font-size: 0.8rem; color: #888;"> /5</span>
                                </div>
                                <div style="flex: 1.5; text-align: center;">
                                    <span class="{row['التصنيف']}">{row['الاتجاه العام']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # إحصائيات الاتجاهات للفقرات
                        high_items = len([i for i in items_trend_data if 'موافق' in i['الاتجاه العام']])
                        neutral_items = len([i for i in items_trend_data if 'محايد' in i['الاتجاه العام']])
                        low_items = len([i for i in items_trend_data if 'غير موافق' in i['الاتجاه العام']])

                        st.markdown(f"""
                        <div class="info-box">
                        <strong>📊 خلاصة الاتجاهات للفقرات:</strong><br>
                        • الفقرات ذات الاتجاه الإيجابي (موافق/موافق بشدة): {high_items} فقرة<br>
                        • الفقرات ذات الاتجاه المحايد: {neutral_items} فقرة<br>
                        • الفقرات ذات الاتجاه السلبي (غير موافق/غير موافق بشدة): {low_items} فقرة
                        </div>
                        """, unsafe_allow_html=True)

                    # ==================== الجزء 3: اختبارات التوزيع الطبيعي ====================
                    st.markdown('<div class="section-header">📈 الجزء 3: اختبارات التوزيع الطبيعي</div>',
                                unsafe_allow_html=True)

                    st.markdown("""
                    <div class="analysis-box">
                    <strong>📈 تحليل النتائج - اختبار Shapiro-Wilk:</strong><br>
                    يتحقق هذا الاختبار من افتراض التوزيع الطبيعي للبيانات.
                    </div>
                    """, unsafe_allow_html=True)

                    normality_data = []
                    for factor in factors:
                        if factor['questions']:
                            factor_values = df[factor['name']].dropna()
                            if len(factor_values) >= 3:
                                stat_shapiro, p_shapiro = shapiro(factor_values)
                                skewness = factor_values.skew()
                                kurtosis = factor_values.kurtosis()
                                normality_data.append({
                                    'المحور': factor['name'],
                                    'الاختبار المستخدم': 'Shapiro-Wilk',
                                    'قيمة الاختبار (Statistic)': stat_shapiro,
                                    'القيمة الاحتمالية (P-value)': p_shapiro,
                                    'التوزيع الطبيعي': '✅ نعم' if p_shapiro > significance_level else '❌ لا',
                                    'الانحراف': skewness,
                                    'التفرطح': kurtosis
                                })

                    if normality_data:
                        normality_df = pd.DataFrame(normality_data)
                        st.dataframe(normality_df.style.format({
                            'قيمة الاختبار (Statistic)': '{:.4f}',
                            'القيمة الاحتمالية (P-value)': '{:.4f}',
                            'الانحراف': '{:.3f}',
                            'التفرطح': '{:.3f}'
                        }), use_container_width=True)

                    # ==================== الجزء 4: الاتجاه العام للمحاور مع الملخص ====================
                    st.markdown('<div class="section-header">🎯 الجزء 4: الاتجاه العام للمحاور والملخص</div>',
                                unsafe_allow_html=True)

                    st.markdown("""
                    <div class="analysis-box">
                    <strong>📈 تحليل النتائج:</strong><br>
                    يوضح هذا القسم الاتجاه العام لكل محور من المحاور الكامنة، مع ملخص تحليلي شامل لكل محور.
                    </div>
                    """, unsafe_allow_html=True)

                    reliability_data = []
                    for factor in factors:
                        if len(factor['questions']) >= 2:
                            questions_df = df[factor['questions']].dropna()
                            if len(questions_df) > 0:
                                try:
                                    cronbach_alpha = pg.cronbach_alpha(data=questions_df)[0]
                                except:
                                    k = len(factor['questions'])
                                    total_variance = questions_df.sum(axis=1).var()
                                    item_variances = questions_df.var().sum()
                                    cronbach_alpha = (k / (k - 1)) * (
                                                1 - (item_variances / total_variance)) if total_variance > 0 else 0

                                mean_score = df[factor['name']].mean()
                                trend_text, trend_class, trend_level = get_likert_trend(mean_score)

                                if cronbach_alpha >= 0.9:
                                    reliability_text = "ممتاز 🌟"
                                elif cronbach_alpha >= 0.8:
                                    reliability_text = "جيد جداً ✅"
                                elif cronbach_alpha >= 0.7:
                                    reliability_text = "مقبول 📊"
                                elif cronbach_alpha >= 0.6:
                                    reliability_text = "ضعيف ⚠️"
                                else:
                                    reliability_text = "غير مقبول ❌"

                                reliability_data.append({
                                    'المحور': factor['name'],
                                    'ألفا كرونباخ (α)': cronbach_alpha,
                                    'عدد الفقرات': len(factor['questions']),
                                    'المتوسط العام': mean_score,
                                    'الاتجاه العام': trend_text,
                                    'المستوى': trend_level,
                                    'التفسير': reliability_text,
                                    'trend_class': trend_class,
                                    'الفقرة': factor['questions']
                                })

                    if reliability_data:
                        # عرض بطاقات المحاور مع الملخص
                        for item in reliability_data:
                            summary = generate_factor_summary(
                                item['المحور'],
                                item['المتوسط العام'],
                                item['الاتجاه العام'],
                                item['المستوى'],
                                item['ألفا كرونباخ (α)'],
                                item['الفقرة']
                            )

                            st.markdown(f"""
                            <div class="factor-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h3 style="margin: 0; color: #2e86ab;">📊 {item['المحور']}</h3>
                                    <span class="{item['trend_class']}" style="font-size: 1.1rem; padding: 5px 15px;">{item['الاتجاه العام']}</span>
                                </div>
                                <div style="display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap;">
                                    <div style="background: #e3f2fd; padding: 10px 15px; border-radius: 10px;">
                                        <strong>المتوسط:</strong> {item['المتوسط العام']:.2f} / 5
                                    </div>
                                    <div style="background: #e8f5e9; padding: 10px 15px; border-radius: 10px;">
                                        <strong>ألفا كرونباخ:</strong> {item['ألفا كرونباخ (α)']:.3f}
                                    </div>
                                    <div style="background: #fff3e0; padding: 10px 15px; border-radius: 10px;">
                                        <strong>عدد الفقرات:</strong> {item['عدد الفقرات']}
                                    </div>
                                    <div style="background: #fce4ec; padding: 10px 15px; border-radius: 10px;">
                                        <strong>الثبات:</strong> {item['التفسير']}
                                    </div>
                                </div>
                                <div class="analysis-box" style="margin: 10px 0 0 0;">
                                    <strong>📝 ملخص المحور:</strong><br>
                                    {summary}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # إحصائيات عامة للمحاور
                        high_factors = len([r for r in reliability_data if r['المستوى'] in ['مرتفع جداً', 'مرتفع']])
                        medium_factors = len([r for r in reliability_data if r['المستوى'] == 'متوسط'])
                        low_factors = len([r for r in reliability_data if r['المستوى'] in ['منخفض', 'منخفض جداً']])

                        st.markdown(f"""
                        <div class="success-box">
                        <strong>📊 الخلاصة العامة للمحاور:</strong><br>
                        • عدد المحاور ذات الاتجاه الإيجابي: {high_factors} محور<br>
                        • عدد المحاور ذات الاتجاه المتوسط: {medium_factors} محور<br>
                        • عدد المحاور ذات الاتجاه السلبي: {low_factors} محور
                        </div>
                        """, unsafe_allow_html=True)

                    # ==================== التوصيات الإحصائية ====================
                    st.markdown('<div class="section-header">💡 التوصيات الإحصائية</div>', unsafe_allow_html=True)

                    recommendations = []

                    if reliability_data:
                        low_reliability = [r['المحور'] for r in reliability_data if r['ألفا كرونباخ (α)'] < 0.7]
                        if low_reliability:
                            recommendations.append(
                                f"⚠️ **تحذير**: المحاور التالية ذات ثبات منخفض: {', '.join(low_reliability)}. يُنصح بمراجعة فقراتها.")

                    if normality_data:
                        non_normal = [n['المحور'] for n in normality_data if n['التوزيع الطبيعي'] == '❌ لا']
                        if non_normal:
                            recommendations.append(
                                f"📊 **توصية**: المتغيرات التالية لا تتبع التوزيع الطبيعي: {', '.join(non_normal)}. يُنصح باستخدام الاختبارات اللامعلمية.")

                    if len(recommendations) == 0:
                        recommendations.append("✅ **ممتاز**: جميع المحاور تظهر ثباتاً جيداً وتتبع التوزيع الطبيعي.")

                    for rec in recommendations:
                        st.markdown(f'<div class="info-box">{rec}</div>', unsafe_allow_html=True)

                    # ==================== الجزء 5: العلاقات مع المتغيرات الاجتماعية ====================
                    st.markdown('<div class="section-header">👥 الجزء 5: تحليل العلاقات مع المتغيرات الاجتماعية</div>',
                                unsafe_allow_html=True)

                    st.markdown("""
                    <div class="analysis-box">
                    <strong>📈 تحليل النتائج:</strong><br>
                    يتم استخدام الاختبارات المناسبة (T-test، ANOVA، Mann-Whitney U، Kruskal-Wallis) لمقارنة متوسطات المحاور بين فئات المتغيرات الاجتماعية.
                    </div>
                    """, unsafe_allow_html=True)

                    social_analysis_data = []

                    for factor in factors:
                        if factor['questions']:
                            all_rel_vars = factor['social_vars'] + factor['independent_vars']
                            for var in all_rel_vars:
                                if var in df.columns:
                                    unique_values = df[var].nunique()
                                    factor_values = df[factor['name']].dropna()

                                    if len(factor_values) >= 3:
                                        _, p_norm = shapiro(factor_values)
                                        is_normal = p_norm > significance_level
                                    else:
                                        is_normal = False

                                    if use_nonparametric or not is_normal:
                                        if unique_values == 2:
                                            unique_vals = df[var].dropna().unique()
                                            if len(unique_vals) == 2:
                                                g1 = df[df[var] == unique_vals[0]][factor['name']].dropna()
                                                g2 = df[df[var] == unique_vals[1]][factor['name']].dropna()
                                                if len(g1) > 0 and len(g2) > 0:
                                                    stat, p_value = mannwhitneyu(g1, g2, alternative='two-sided')
                                                    test_name = "Mann-Whitney U"
                                                    test_description = "اختبار مان Whitney - مقارنة مجموعتين مستقلتين (لامعلمي)"
                                                else:
                                                    continue
                                            else:
                                                continue
                                        elif unique_values > 2:
                                            groups = [df[df[var] == val][factor['name']].dropna() for val in
                                                      df[var].dropna().unique()]
                                            groups = [g for g in groups if len(g) > 0]
                                            if len(groups) > 1:
                                                stat, p_value = kruskal(*groups)
                                                test_name = "Kruskal-Wallis"
                                                test_description = "اختبار كروسكال واليس - مقارنة أكثر من مجموعتين (لامعلمي)"
                                            else:
                                                continue
                                        else:
                                            continue
                                    else:
                                        if unique_values == 2:
                                            unique_vals = df[var].dropna().unique()
                                            if len(unique_vals) == 2:
                                                g1 = df[df[var] == unique_vals[0]][factor['name']].dropna()
                                                g2 = df[df[var] == unique_vals[1]][factor['name']].dropna()
                                                if len(g1) > 0 and len(g2) > 0:
                                                    stat, p_value = ttest_ind(g1, g2, equal_var=False)
                                                    test_name = "T-test"
                                                    test_description = "اختبار T-test - مقارنة مجموعتين مستقلتين (معلمي)"
                                                else:
                                                    continue
                                            else:
                                                continue
                                        elif unique_values > 2:
                                            groups = [df[df[var] == val][factor['name']].dropna() for val in
                                                      df[var].dropna().unique()]
                                            groups = [g for g in groups if len(g) > 0]
                                            if len(groups) > 1:
                                                stat, p_value = f_oneway(*groups)
                                                test_name = "ANOVA"
                                                test_description = "تحليل التباين الأحادي ANOVA - مقارنة أكثر من مجموعتين (معلمي)"
                                            else:
                                                continue
                                        else:
                                            continue

                                    social_analysis_data.append({
                                        'المحور': factor['name'],
                                        'المتغير': var,
                                        'الاختبار المستخدم': test_name,
                                        'وصف الاختبار': test_description,
                                        'قيمة الاختبار': stat,
                                        'القيمة الاحتمالية (P-value)': p_value,
                                        'الدلالة': '✅ دالة' if p_value < significance_level else '❌ غير دالة'
                                    })

                    if social_analysis_data:
                        social_analysis_df = pd.DataFrame(social_analysis_data)
                        st.dataframe(social_analysis_df.style.format({
                            'قيمة الاختبار': '{:.4f}',
                            'القيمة الاحتمالية (P-value)': '{:.4f}'
                        }), use_container_width=True)

                        if show_boxplots:
                            significant_results = social_analysis_df[social_analysis_df['الدلالة'] == '✅ دالة']
                            if len(significant_results) > 0:
                                st.subheader("📦 المخططات الصندوقية للعلاقات الدالة إحصائياً")
                                for _, row in significant_results.iterrows():
                                    fig = px.box(df, x=row['المتغير'], y=row['المحور'],
                                                 title=f"العلاقة بين {row['المحور']} و {row['المتغير']} - باستخدام {row['الاختبار المستخدم']}",
                                                 color=row['المتغير'])
                                    st.plotly_chart(fig, use_container_width=True)

                    # ==================== الجزء 6: تحليل الارتباط ====================
                    if show_correlation:
                        st.markdown('<div class="section-header">🔗 الجزء 6: تحليل الارتباط بين المتغيرات</div>',
                                    unsafe_allow_html=True)

                        st.markdown("""
                        <div class="analysis-box">
                        <strong>📈 تحليل النتائج - معامل ارتباط بيرسون (Pearson Correlation):</strong><br>
                        يقيس قوة واتجاه العلاقة الخطية بين المتغيرات الرقمية.
                        </div>
                        """, unsafe_allow_html=True)

                        factor_columns = [f['name'] for f in factors if f['questions']]
                        all_corr_vars = factor_columns + independent_vars + social_vars
                        numeric_columns = df[all_corr_vars].select_dtypes(include=[np.number]).columns.tolist()

                        if len(numeric_columns) > 1:
                            corr_matrix = df[numeric_columns].corr()
                            if show_heatmap:
                                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                                color_continuous_scale='RdBu_r',
                                                title="مصفوفة الارتباط - Pearson Correlation",
                                                zmin=-1, zmax=1)
                                st.plotly_chart(fig, use_container_width=True)

                            corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i + 1, len(corr_matrix.columns)):
                                    corr_value = corr_matrix.iloc[i, j]
                                    if abs(corr_value) > 0.5:
                                        corr_pairs.append({
                                            'المتغير 1': corr_matrix.columns[i],
                                            'المتغير 2': corr_matrix.columns[j],
                                            'معامل الارتباط (r)': corr_value,
                                            'الاختبار': 'Pearson',
                                            'الدلالة': '✅ دالة' if abs(corr_value) > 0.7 else '⚠️ متوسطة'
                                        })

                            if corr_pairs:
                                corr_pairs_df = pd.DataFrame(corr_pairs)
                                st.dataframe(corr_pairs_df.sort_values('معامل الارتباط (r)', key=abs, ascending=False),
                                             use_container_width=True)

                    # ==================== الجزء 7: تحليل الانحدار المتقدم ====================
                    model_sm = None
                    if show_regression and independent_factors and dependent_factor:
                        st.markdown('<div class="section-header">📈 الجزء 7: تحليل الانحدار المتقدم</div>',
                                    unsafe_allow_html=True)

                        st.markdown("""
                        <div class="analysis-box">
                        <strong>📈 تحليل النتائج - الانحدار الخطي المتعدد (Multiple Linear Regression):</strong><br>
                        يحدد تأثير المتغيرات المستقلة على المتغير التابع.
                        </div>
                        """, unsafe_allow_html=True)

                        try:
                            X = df[independent_factors]
                            y = df[dependent_factor]
                            regression_data = pd.concat([X, y], axis=1).dropna()
                            X_clean = regression_data[independent_factors]
                            y_clean = regression_data[dependent_factor]

                            if len(regression_data) > 0:
                                X_sm = sm.add_constant(X_clean)
                                model_sm = sm.OLS(y_clean, X_sm).fit()

                                st.subheader("📊 نتائج تحليل الانحدار")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**الاختبار:** Multiple Linear Regression")
                                    st.write(f"**R² (معامل التحديد) = {model_sm.rsquared:.4f}**")
                                    st.write(f"**R² المعدل = {model_sm.rsquared_adj:.4f}**")
                                    st.write(f"**F-test = {model_sm.fvalue:.4f} (p = {model_sm.f_pvalue:.4f})**")

                                with col2:
                                    coef_df = pd.DataFrame({
                                        'المتغير': model_sm.params.index,
                                        'المعامل (B)': model_sm.params.values,
                                        'الخطأ المعياري': model_sm.bse.values,
                                        't-test': model_sm.tvalues.values,
                                        'P-value': model_sm.pvalues.values
                                    })
                                    st.dataframe(coef_df.style.format({
                                        'المعامل (B)': '{:.4f}',
                                        'الخطأ المعياري': '{:.4f}',
                                        't-test': '{:.4f}',
                                        'P-value': '{:.4f}'
                                    }), use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ خطأ في تحليل الانحدار: {e}")

                    # ==================== الجزء 8: التحليل العاملي الاستكشافي ====================
                    if show_efa and len(question_vars) >= 3:
                        st.markdown('<div class="section-header">🔍 الجزء 8: التحليل العاملي الاستكشافي (EFA)</div>',
                                    unsafe_allow_html=True)

                        st.markdown("""
                        <div class="analysis-box">
                        <strong>📈 تحليل النتائج - التحليل العاملي الاستكشافي (Factor Analysis):</strong><br>
                        يختبر البنية العاملية للفقرات ويقترح تجميعها.
                        </div>
                        """, unsafe_allow_html=True)

                        try:
                            from sklearn.decomposition import FactorAnalysis
                            from sklearn.preprocessing import StandardScaler

                            efa_data = df[question_vars].dropna().astype(float)

                            if len(efa_data) >= 5:
                                scaler = StandardScaler()
                                efa_scaled = scaler.fit_transform(efa_data)

                                n_components = min(num_factors, len(question_vars))
                                fa = FactorAnalysis(n_components=n_components, random_state=42)
                                fa_result = fa.fit_transform(efa_scaled)

                                st.write(f"**الاختبار المستخدم:** Factor Analysis")
                                st.write(f"**التباين المفسر:** {fa.noise_variance_:.4f}")

                                loadings_df = pd.DataFrame(
                                    fa.components_.T,
                                    columns=[f"العامل_{i + 1}" for i in range(fa.components_.shape[0])],
                                    index=question_vars
                                )
                                st.dataframe(loadings_df.style.format('{:.3f}'), use_container_width=True)
                            else:
                                st.warning("⚠️ عدد البيانات غير كافٍ للتحليل العاملي")

                        except Exception as e:
                            st.error(f"❌ خطأ في التحليل العاملي: {e}")

                    # ==================== الجزء 9: تحليل التجميع (بدون إحصائيات المجموعات) ====================
                    if show_clustering and len(factor_columns) >= 2:
                        st.markdown('<div class="section-header">🎯 الجزء 9: تحليل التجميع (Cluster Analysis)</div>',
                                    unsafe_allow_html=True)

                        st.markdown("""
                        <div class="analysis-box">
                        <strong>📈 تحليل النتائج - تحليل التجميع (K-Means Clustering):</strong><br>
                        يقسم المشاركين إلى مجموعات متجانسة بناءً على استجاباتهم.
                        </div>
                        """, unsafe_allow_html=True)

                        try:
                            from sklearn.cluster import KMeans
                            from sklearn.metrics import silhouette_score

                            cluster_data = df[factor_columns].dropna()

                            if len(cluster_data) >= 5:
                                scaler = StandardScaler()
                                cluster_scaled = scaler.fit_transform(cluster_data)

                                inertias = []
                                silhouette_scores = []
                                max_k = min(8, len(cluster_data) - 1)

                                if max_k >= 2:
                                    k_range = range(2, max_k + 1)
                                    for k in k_range:
                                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                        kmeans.fit(cluster_scaled)
                                        inertias.append(kmeans.inertia_)
                                        silhouette_scores.append(silhouette_score(cluster_scaled, kmeans.labels_))

                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                                    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
                                    ax1.set_xlabel('Number of Clusters (k)')
                                    ax1.set_ylabel('Inertia')
                                    ax1.set_title('Elbow Method - K-Means')
                                    ax1.grid(True, alpha=0.3)
                                    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
                                    ax2.set_xlabel('Number of Clusters (k)')
                                    ax2.set_ylabel('Silhouette Score')
                                    ax2.set_title('Silhouette Analysis')
                                    ax2.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    st.pyplot(fig)

                                    best_k = k_range[np.argmax(silhouette_scores)]
                                    st.info(
                                        f"✨ **العدد الأمثل للمجموعات:** {best_k} (Silhouette Score: {max(silhouette_scores):.3f})")

                                    st.markdown("""
                                    <div class="info-box">
                                    <strong>📌 ملاحظة:</strong><br>
                                    تم تحديد العدد الأمثل للمجموعات بناءً على معامل الصورة الظلية (Silhouette Score).<br>
                                    يمكن استخدام هذه النتيجة لتقسيم العينة إلى شرائح متجانسة في التحليلات اللاحقة.
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ عدد البيانات غير كافٍ")
                            else:
                                st.warning("⚠️ عدد البيانات غير كافٍ لتحليل التجميع (يحتاج إلى 5 عينات على الأقل)")
                        except Exception as e:
                            st.error(f"❌ خطأ في تحليل التجميع: {e}")

                    # ==================== الجزء 10: تقرير ملخص ====================
                    st.markdown('<div class="section-header">📑 الجزء 10: التقرير النهائي والملخص</div>',
                                unsafe_allow_html=True)

                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ اكتمل التحليل الإحصائي المتقدم بنجاح!</strong><br><br>
                        <strong>ملخص النتائج:</strong><br>
                    """, unsafe_allow_html=True)

                    st.write(f"- **عدد المتغيرات المحللة:** {len(all_columns)}")
                    st.write(f"- **عدد المحاور الكامنة:** {len([f for f in factors if f['questions']])}")
                    st.write(f"- **عدد المتغيرات الاجتماعية:** {len(social_vars)}")
                    st.write(f"- **عدد المتغيرات المستقلة:** {len(independent_vars)}")

                    if reliability_data:
                        avg_alpha = np.mean([r['ألفا كرونباخ (α)'] for r in reliability_data])
                        st.write(f"- **متوسط معامل ألفا كرونباخ:** {avg_alpha:.3f}")

                    if social_analysis_data:
                        sig_count = len([r for r in social_analysis_data if r['الدلالة'] == '✅ دالة'])
                        st.write(f"- **العلاقات الدالة إحصائياً:** {sig_count} من {len(social_analysis_data)}")

                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button("📥 تحميل التقرير الكامل (Excel)", type="secondary"):
                        try:
                            with pd.ExcelWriter('تقرير_التحليل_الإحصائي.xlsx') as writer:
                                if 'desc_stats' in locals() and desc_stats is not None:
                                    desc_stats.to_excel(writer, sheet_name='الإحصاءات_الوصفية')
                                if 'normality_df' in locals() and normality_df is not None:
                                    normality_df.to_excel(writer, sheet_name='اختبارات_الطبيعية')
                                if 'reliability_data' in locals() and reliability_data:
                                    pd.DataFrame(reliability_data).to_excel(writer, sheet_name='الاتجاه_و_الثبات')
                                if 'social_analysis_df' in locals() and social_analysis_df is not None:
                                    social_analysis_df.to_excel(writer, sheet_name='التحليل_الاجتماعي')
                            st.success("✅ تم حفظ التقرير بنجاح!")
                        except Exception as e:
                            st.error(f"❌ خطأ في حفظ التقرير: {e}")

else:
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>🎯 مرحباً بك في نظام التحليل الإحصائي المتقدم للاستبيانات</h2>
        <p style='font-size: 1.2rem; color: #666; margin-top: 20px;'>
            نظام متكامل لتحليل بيانات الاستبيانات مع دعم للاختبارات الإحصائية المتقدمة
        </p>
    </div>
    """, unsafe_allow_html=True)