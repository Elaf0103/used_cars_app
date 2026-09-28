# ---------------------------------------------------------------
# Analysis of Factors Affecting Used Car Prices in Saudi Arabia
# A bilingual (English / Arabic) single-page Streamlit app with two tabs:
#   1) Price Estimator: the data at a glance, then the estimator itself
#   2) Analysis:        the 4 charts (each paired with its takeaway),
#                        a "what drives the price" summary, and a
#                        "Methodology & Key Decisions" accordion
#
# Every visible string lives in the TEXT dictionary below, keyed by a short
# identifier with an "en" and an "ar" value. The chosen language is stored in
# st.session_state["lang"]; picking Arabic also switches the page to RTL.
# Run with:  streamlit run app.py
# ---------------------------------------------------------------

from html import escape            # makes text safe to put inside HTML

import joblib                      # loads the saved model files (.pkl)
import numpy as np                 # used for np.exp() and creating zeros
import pandas as pd                # tables (DataFrames)
import plotly.express as px        # simple interactive charts
import streamlit as st             # the web app framework

# ---------------------------------------------------------------
# Page settings (must be the first Streamlit command in the script)
# "wide" lets the page use more of the screen, like a dashboard.
# The browser-tab title is static, so it carries both languages.
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Prices in Saudi Arabia | أسعار السيارات المستعملة في السعودية",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------
# Color palette (used by the CSS below and by the charts)
# ---------------------------------------------------------------
PAGE_BACKGROUND = "#F7FBFE"   # almost-white with a faint blue tint
ACCENT = "#6EC1E4"            # calm sky blue: buttons, bars, lines, highlights
ACCENT_HOVER = "#55B0D8"      # slightly darker sky blue for hover states
ACCENT_SOFT = "#EAF6FC"       # very pale blue for backgrounds and "folder tab" flourishes
CARD_BORDER = "#DCEEF9"       # pale blue border for cards
TEXT_COLOR = "#2C3E50"        # dark slate gray (softer than pure black)
MUTED_TEXT = "#5D7285"        # lighter slate for captions and small labels
GRID_COLOR = "#E8F3FA"        # very faint grid lines inside charts

# Model quality numbers, measured on the 20% test set in the Colab notebook
# (final model: Make + Type + Year + Mileage, trained on log(Price))
MODEL_R2 = 0.83
MODEL_TYPICAL_ERROR_SAR = 14686          # the MAE (mean absolute error)
TEST_SET_SIZE = 1096

# Data-cleaning numbers from the notebook: how many raw listings there were,
# and how many rows each cleaning step removed (see the FAQ for the reasoning).
RAW_LISTINGS = 8248
REMOVED_NEGOTIABLE = 2592
REMOVED_DUPLICATES = 20
REMOVED_IMPLAUSIBLE = 156        # zero price, pre-1990, >600k km, tiny modern prices, bad mileage fixes


# ---------------------------------------------------------------
# Every piece of visible text, in both languages.
# L("English", "العربية") builds one entry. Placeholders such as {rows:,} are
# filled in by t() below. Numbers stay in Western digits (0-9) in both
# languages so prices, charts and text always match each other; the currency
# reads "SAR" in English and "ريال" / "ريال سعودي" in Arabic.
# ---------------------------------------------------------------
def L(en: str, ar: str) -> dict:
    return {"en": en, "ar": ar}


TEXT = {
    # ---- Tabs and general ----
    "tab_estimator": L("Price Estimator", "مُقدِّر السعر"),
    "tab_analysis": L("Analysis", "التحليل"),
    "currency": L("SAR", "ريال"),
    "currency_long": L("SAR", "ريال سعودي"),
    "km": L("km", "كم"),

    # ---- Estimator tab: data at a glance ----
    "glance_title": L("Used Cars in Saudi Arabia, at a glance", "السيارات المستعملة في السعودية: لمحة سريعة"),
    "glance_sub": L(
        "A quick look at the data the estimator learned from, and how much to trust it, "
        "then try it on your own car below.",
        "نظرة سريعة على البيانات التي تعلّم منها المُقدِّر ومدى الثقة بها، ثم جرّب المُقدِّر على سيارتك في الأسفل.",
    ),
    "byline": L("By Elaf Alyoubi", "إعداد: {name}"),
    "stat_rows_label": L("Cars after cleaning", "السيارات بعد التنظيف"),
    "stat_rows_note": L("listings kept for the analysis", "إعلان بقي للتحليل"),
    "stat_cols_label": L("Columns", "الأعمدة"),
    "stat_median_label": L("Median price", "السعر الوسيط"),
    "stat_median_note": L("half of the cars cost less than this", "نصف السيارات أرخص من هذا السعر"),
    "stat_make_label": L("Most common make", "أكثر ماركة شيوعًا"),
    "stat_make_note": L("{n:,} listings ({p} of all cars)", "{n:,} إعلان ({p} من السيارات)"),
    "stat_year_label": L("Most common year", "أكثر سنة صنع شيوعًا"),
    "stat_year_note": L("{n:,} cars from this year", "{n:,} سيارة من هذه السنة"),
    "stat_range_label": L("Price range", "نطاق الأسعار"),
    "stat_range_note": L("SAR, from cheapest to most expensive", "ريال سعودي، من الأرخص إلى الأغلى"),
    "col_Make": L("Make", "الماركة"),
    "col_Type": L("Type", "الطراز"),
    "col_Year": L("Year", "سنة الصنع"),
    "col_Mileage": L("Mileage", "المسافة المقطوعة"),
    "col_Region": L("Region", "المنطقة"),
    "col_Price": L("Price", "السعر"),

    # ---- Estimator tab: trust cards ----
    "dq_label": L("Data quality", "جودة البيانات"),
    "dq_headline": L(
        "<b>{raw:,}</b> raw listings → <b>{rows:,}</b> kept",
        "<b>{raw:,}</b> إعلان خام ← <b>{rows:,}</b> إعلان بعد التنظيف",
    ),
    "dq_kept": L("<b>{p}</b> kept", "<b>{p}</b> محتفَظ بها"),
    "dq_removed": L("<b>{n:,}</b> removed", "<b>{n:,}</b> محذوفة"),
    "dq_text": L(
        "Most of the cut ({neg:,} listings) were priced “Negotiable” instead of a number. "
        "The rest were {dup} duplicates and {imp} rows with impossible values.",
        "معظم المحذوف ({neg:,} إعلان) كان سعره «قابل للتفاوض» بدلًا من رقم. "
        "والباقي {dup} إعلانًا مكرّرًا و{imp} صفًّا بقيم غير منطقية.",
    ),
    "mc_label": L("Model confidence", "ثقة النموذج"),
    "mc_r2": L("R²: share of price variation explained", "R²: نسبة تباين الأسعار التي يفسّرها النموذج"),
    "mc_mae": L("MAE: average error on unseen cars", "MAE: متوسط الخطأ على سيارات لم يرَها النموذج"),
    "mc_text": L(
        "Measured on {n:,} cars the model never saw during training. Full details under "
        "“Methodology & Key Decisions” on the Analysis tab.",
        "قِيس الأداء على {n:,} سيارة لم يرَها النموذج أثناء التدريب. التفاصيل الكاملة ضمن "
        "«المنهجية والقرارات الأساسية» في تبويب التحليل.",
    ),

    # ---- Estimator tab: the tool ----
    "hero_pill": L("Try it yourself", "جرّبه بنفسك"),
    "hero_title": L("Get a price estimate", "احصل على تقدير للسعر"),
    "hero_lead": L(
        "Answer four quick questions and get an instant, data-backed price estimate, "
        "built from {n:,} real listings from across Saudi Arabia.",
        "أجب عن أربعة أسئلة سريعة لتحصل فورًا على تقدير للسعر مبني على البيانات، "
        "اعتمادًا على {n:,} إعلان حقيقي من مختلف مناطق المملكة.",
    ),
    "inputs_title": L("Car details", "بيانات السيارة"),
    "f_make": L("Make", "الماركة"),
    "f_type": L("Type / Model", "الطراز / الموديل"),
    "f_year": L("Production year", "سنة الصنع"),
    "f_mileage": L("Mileage (km)", "المسافة المقطوعة (كم)"),
    "btn_predict": L("Predict Price", "قدّر السعر"),
    "ph_title": L("Your estimate will appear here", "سيظهر تقديرك هنا"),
    "ph_text": L(
        "Pick a car on the left and press Predict Price.",
        "اختر سيارتك ثم اضغط «قدّر السعر».",
    ),
    "warn_missing": L(
        "No {make} {type} exists in the data, so a reliable prediction is not possible.",
        "لا توجد في البيانات سيارة {make} {type}، لذا لا يمكن تقديم تقدير موثوق.",
    ),
    "res_label": L("Estimated price", "السعر التقديري"),
    "res_reliable_h": L("How reliable is this?", "ما مدى موثوقية هذا التقدير؟"),
    "res_reliable_t": L(
        "This model explains about {r2p} of price variation (R² = {r2:.2f}), with a typical "
        "prediction error of around {mae:,} SAR.",
        "يفسّر هذا النموذج نحو {r2p} من تباين الأسعار (R² = {r2:.2f})، "
        "بمتوسط خطأ في التنبؤ يبلغ نحو {mae:,} ريال سعودي.",
    ),
    "res_why_h": L("Why this price?", "لماذا هذا السعر؟"),

    # ---- "Why this price?" explanation: sentence templates (not word-for-word) ----
    "desc_open_up": L(
        "At {p} SAR, this {make} lands well above the brand's usual {tp} SAR, and the "
        "numbers explain why.",
        "بسعر {p} ريال، تأتي سيارة {make} هذه أعلى بوضوح من السعر المعتاد للماركة ({tp} ريال)، "
        "والأرقام تفسّر السبب.",
    ),
    "desc_open_dn": L(
        "At {p} SAR, this {make} comes in well under the brand's usual {tp} SAR.",
        "بسعر {p} ريال، تأتي سيارة {make} هذه أقل بوضوح من السعر المعتاد للماركة ({tp} ريال).",
    ),
    "desc_open_mid": L(
        "At {p} SAR, this {make} lands right around the brand's usual {tp} SAR.",
        "بسعر {p} ريال، تقع سيارة {make} هذه قرب السعر المعتاد للماركة ({tp} ريال).",
    ),
    "desc_m_m": L(
        "Year ({year}) and mileage ({km} km) are both bang on the {make} norm, so most of "
        "this number is really coming from the {type} trim itself",
        "سنة الصنع ({year}) والمسافة المقطوعة ({km} كم) قريبتان من المعتاد لماركة {make}، "
        "لذا يعود معظم هذا السعر إلى الطراز {type} نفسه",
    ),
    "desc_u_u": L(
        "A {year} build with only {km} km on the clock beats the typical {ty}, {tk} km {make} "
        "on both counts: younger and gentler mileage are stacking in this car's favor",
        "سيارة موديل {year} لم تقطع سوى {km} كم تتفوّق على سيارة {make} المعتادة "
        "(موديل {ty}، {tk} كم) من الناحيتين، فحداثتها وقلّة استخدامها يعملان معًا لصالحها",
    ),
    "desc_d_d": L(
        "At {year} with {km} km already covered, it's older and more traveled than the typical "
        "{ty}, {tk} km {make}, and both of those are working against the price",
        "بموديل {year} وقد قطعت {km} كم، فهي أقدم وأكثر استخدامًا من سيارة {make} المعتادة "
        "(موديل {ty}، {tk} كم)، وكلا العاملين يضغطان على السعر نحو الأسفل",
    ),
    "desc_u_m": L(
        "Being a {year} model puts it ahead of the typical {ty} {make}, while its mileage "
        "({km} km) is close to par, so the age is doing most of the talking here",
        "كونها موديل {year} يضعها في مرتبة أفضل من سيارة {make} المعتادة (موديل {ty})، "
        "بينما مسافتها المقطوعة ({km} كم) قريبة من المعتاد، فالحداثة هي العامل الأبرز هنا",
    ),
    "desc_d_m": L(
        "At {year}, it trails the typical {ty} {make}, while its mileage ({km} km) is close "
        "to par, so the age is what's really pulling the price down",
        "بموديل {year} تتأخر عن سيارة {make} المعتادة (موديل {ty})، بينما مسافتها المقطوعة "
        "({km} كم) قريبة من المعتاد، فالعمر هو ما يسحب السعر نحو الأسفل",
    ),
    "desc_m_u": L(
        "The mileage is the real story here: {km} km against a typical {tk} km, while the "
        "{year} build year is close to par for the brand",
        "المسافة المقطوعة هي الأبرز هنا: {km} كم مقابل {tk} كم للسيارة المعتادة، "
        "بينما سنة الصنع ({year}) قريبة من المعتاد للماركة",
    ),
    "desc_m_d": L(
        "The mileage is the real story here: {km} km against a typical {tk} km, and it's "
        "dragging the price down even though the {year} build year is close to par",
        "المسافة المقطوعة هي الأبرز هنا: {km} كم مقابل {tk} كم للسيارة المعتادة، "
        "وهي تسحب السعر نحو الأسفل رغم أن سنة الصنع ({year}) قريبة من المعتاد للماركة",
    ),
    "desc_u_d": L(
        "It's newer than the typical {ty} {make}, which helps its case, but {km} km on the "
        "odometer (versus a typical {tk} km) is pulling the other way",
        "هي أحدث من سيارة {make} المعتادة (موديل {ty})، وهذا في صالحها، لكن {km} كم على العدّاد "
        "(مقابل {tk} كم للمعتادة) تسحب السعر في الاتجاه المعاكس",
    ),
    "desc_d_u": L(
        "It's older than the typical {ty} {make}, which works against it, but the light "
        "mileage, just {km} km against a typical {tk} km, is clawing some value back",
        "هي أقدم من سيارة {make} المعتادة (موديل {ty})، وهذا يعمل ضدها، لكن قلّة المسافة "
        "المقطوعة، {km} كم فقط مقابل {tk} كم للمعتادة، تعوّض جزءًا من القيمة",
    ),
    "desc_extra": L(
        "; the {type} trim is clearly doing a lot of the rest of the work",
        "؛ ويبدو أن الطراز {type} يفسّر الجزء الأكبر من الفرق المتبقي",
    ),
    "desc_disclaimer": L(
        "Just keep in mind this comes from a simple Linear Regression using only make, "
        "model, year and mileage: a smart starting point, not the final word.",
        "ولا تنسَ أن هذا التقدير مبني على انحدار خطي بسيط يستخدم الماركة والطراز وسنة الصنع "
        "والمسافة المقطوعة فقط، فهو نقطة انطلاق جيدة وليس الكلمة الأخيرة.",
    ),

    # ---- Analysis tab: header, charts, takeaways ----
    "an_title": L("Analysis", "التحليل"),
    "an_sub": L(
        "These are the exact patterns the prediction model learned from. Each chart "
        "is paired with its takeaway, and the methodology notes at the end explain the "
        "decisions behind the project.",
        "هذه هي الأنماط نفسها التي تعلّم منها نموذج التنبؤ. كل رسم بياني مقترن بخلاصته، "
        "وتشرح ملاحظات المنهجية في الختام القرارات التي قام عليها المشروع.",
    ),
    "insight_label": L("Key takeaway", "الخلاصة الرئيسية"),
    "ch_price_title": L("Price distribution", "توزيع الأسعار"),
    "ch_price_x": L("Price (SAR)", "السعر (ريال سعودي)"),
    "ch_price_y": L("Number of cars", "عدد السيارات"),
    "ch_price_take": L(
        "About {p} of cars are priced under 100,000 SAR; a few very expensive cars stretch "
        "the chart to the right.",
        "نحو {p} من السيارات يقل سعرها عن 100,000 ريال؛ وتمدّد قلّة من السيارات باهظة الثمن "
        "الرسم نحو اليمين.",
    ),
    "ch_make_title": L("Median price of the most common makes", "السعر الوسيط لأكثر الماركات شيوعًا"),
    "ch_make_x": L("Median price (SAR)", "السعر الوسيط (ريال سعودي)"),
    "ch_make_take": L(
        "{top} is the most expensive of these makes, about {ratio:.1f} times the price of "
        "{low}, the cheapest. This is why Make moves the prediction so much.",
        "{top} هي الأغلى بين هذه الماركات، بنحو {ratio:.1f} ضعف سعر {low} الأرخص، "
        "ولهذا تؤثر الماركة كثيرًا في التقدير.",
    ),
    "ch_year_title": L("Median price by production year", "السعر الوسيط حسب سنة الصنع"),
    "ch_year_x": L("Production year", "سنة الصنع"),
    "ch_year_y": L("Median price (SAR)", "السعر الوسيط (ريال سعودي)"),
    "ch_year_take": L(
        "Newer cars are worth more: the median price is about {a:,.0f} SAR for 2010 models "
        "and {b:,.0f} SAR for 2020 models. This is the same effect behind the model's Year input.",
        "السيارات الأحدث أعلى قيمة: السعر الوسيط نحو {a:,.0f} ريال لموديلات 2010 و{b:,.0f} ريال "
        "لموديلات 2020، وهذا هو الأثر نفسه وراء مُدخل سنة الصنع في النموذج.",
    ),
    "ch_km_title": L("Price vs. mileage", "السعر مقابل المسافة المقطوعة"),
    "ch_km_x": L("Mileage (km)", "المسافة المقطوعة (كم)"),
    "ch_km_y": L("Price (SAR)", "السعر (ريال سعودي)"),
    "ch_km_take": L(
        "Each dot is one car. Prices tend to fall as mileage rises. It's the same downward "
        "pull you'll see in the Mileage part of any prediction on the Price Estimator tab.",
        "كل نقطة تمثّل سيارة واحدة. يميل السعر إلى الانخفاض مع ارتفاع المسافة المقطوعة، "
        "وهو الأثر نفسه الذي يظهر في جانب المسافة المقطوعة من أي تقدير في تبويب مُقدِّر السعر.",
    ),

    # ---- Analysis tab: what drives the price ----
    "drv_title": L("What actually drives the price?", "ما الذي يحرّك السعر فعلًا؟"),
    "drv_sub": L(
        "Four of these feed the prediction model directly; Region is a real pattern "
        "in the data too, just not one the model currently uses.",
        "أربعة من هذه العوامل يعتمد عليها نموذج التنبؤ مباشرة؛ أما المنطقة فهي نمط حقيقي "
        "في البيانات أيضًا، لكن النموذج لا يستخدمه حاليًا.",
    ),
    "drv_make_label": L("Make", "الماركة"),
    "drv_make_title": L("Luxury brands cost far more", "الماركات الفاخرة أعلى سعرًا بكثير"),
    "drv_make_figure": L(
        "Nissan 37,000 → Rolls-Royce 560,000 SAR",
        "نيسان 37,000 ← رولز رويس 560,000 ريال",
    ),
    "drv_make_note": L(
        "Across 61 makes, the badge on the car sets the baseline price.",
        "عبر 61 ماركة، تحدّد ماركة السيارة السعر الأساسي.",
    ),
    "drv_year_label": L("Year", "سنة الصنع"),
    "drv_year_title": L("Newer models cost more", "الموديلات الأحدث أعلى سعرًا"),
    "drv_year_figure": L("2001: ~13,500 → 2018: 70,000 SAR", "2001: نحو 13,500 ← 2018: 70,000 ريال"),
    "drv_year_note": L(
        "Median price climbs steadily with production year (correlation 0.35).",
        "يرتفع السعر الوسيط باطّراد مع سنة الصنع (معامل الارتباط 0.35).",
    ),
    "drv_mileage_label": L("Mileage", "المسافة المقطوعة"),
    "drv_mileage_title": L("More km, lower price", "كلما زادت الكيلومترات انخفض السعر"),
    "drv_mileage_figure": L(
        "94.2% of 200,000+ km cars ≤ 100,000 SAR",
        "94.2% من السيارات التي تجاوزت 200,000 كم سعرها 100,000 ريال أو أقل",
    ),
    "drv_mileage_note": L(
        "Price falls as the odometer rises (correlation -0.35).",
        "ينخفض السعر مع ارتفاع قراءة العدّاد (معامل الارتباط ‎-0.35).",
    ),
    "drv_type_label": L("Model / Type", "الطراز / الموديل"),
    "drv_type_title": L("Same brand, worlds apart", "ماركة واحدة وفروق شاسعة"),
    "drv_type_figure": L("R² 0.54 → 0.83 once Type is added", "R² من 0.54 إلى 0.83 بعد إضافة الطراز"),
    "drv_type_note": L(
        "The exact model (Corolla vs. Land Cruiser) was the single biggest accuracy gain in the project.",
        "الطراز الدقيق (كورولا مقابل لاند كروزر) كان أكبر تحسّن في الدقة على مستوى المشروع.",
    ),
    "drv_region_label": L("Region", "المنطقة"),
    "drv_region_title": L("Location shifts it too", "الموقع يؤثر أيضًا"),
    "drv_region_figure": L(
        "Dammam 69,000 vs. Makkah 43,000 SAR",
        "الدمام 69,000 مقابل مكة المكرمة 43,000 ريال",
    ),
    "drv_region_note": L(
        "Median prices differ across the 8 largest regions, but the model doesn't use Region.",
        "تختلف الأسعار الوسيطة بين أكبر 8 مناطق، لكن النموذج لا يستخدم المنطقة.",
    ),
    "drv_region_tag": L("Not used by the model", "لا يستخدمها النموذج"),

    # ---- Methodology & Key Decisions (FAQ) ----
    "faq_title": L("Methodology & Key Decisions", "المنهجية والقرارات الأساسية"),
    "faq_sub": L(
        "The reasoning and exact figures behind the model, the data cleaning and the "
        "charts, starting with what matters most for trusting the estimates. Click any "
        "question to open it, and open as many as you like at once.",
        "المبررات والأرقام الدقيقة وراء النموذج وتنظيف البيانات والرسوم البيانية، بدءًا بما هو "
        "أهم للوثوق بالتقديرات. اضغط على أي سؤال لفتحه، ويمكنك فتح ما تشاء منها في الوقت نفسه.",
    ),
    "filter_all": L("All", "الكل"),
    "filter_model": L("Model", "النموذج"),
    "filter_data": L("Data", "البيانات"),
    "filter_support": L("Supporting Details", "تفاصيل داعمة"),
    "about_title": L("About the data", "عن البيانات"),
    "about_text": L(
        "The raw listings come from the Saudi Arabia Used Cars Dataset on Kaggle, published by "
        "Turki Bintaleb and scraped from the Syarah website. The cleaning, the analysis and the "
        "prediction model in this dashboard are this project's own work.",
        "الإعلانات الخام مصدرها مجموعة بيانات السيارات المستعملة في السعودية على Kaggle، التي نشرها "
        "تركي بن طالب وجمعها من موقع سيارة (Syarah). أما التنظيف والتحليل ونموذج التنبؤ في هذه "
        "اللوحة فهي من عمل هذا المشروع.",
    ),
    "faq_count": L(
        "Showing {n} of {m} questions · {group}",
        "عرض {n} من {m} سؤالًا · {group}",
    ),

    "faq_q1": L("How accurate is this model, really?", "ما مدى دقة هذا النموذج فعلًا؟"),
    "faq_a1": L(
        "The final model uses **Make + Type (the specific model) + Year + Mileage** as "
        "inputs and is trained on log(Price). It was tested on **{test:,} cars** "
        "(20% of the {rows:,}) that it never saw during training:\n\n"
        "- **R² = {r2:.2f}**: it explains about {r2p} of the variation in price.\n"
        "- **MAE = {mae:,} SAR**: on average, its estimate is off by "
        "about this much (mean absolute error). For context, the median car in the data "
        "costs 59,000 SAR.\n\n"
        "**Known limitation:** the model does *not* know the car's condition, its trim or "
        "options level (e.g. “Full” vs. “Standard”), its color, or its accident history. "
        "None of these are in the cleaned dataset. So an estimate can be off for a car "
        "that is unusually well-equipped, or damaged, for its make, year and mileage.",
        "يستخدم النموذج النهائي **الماركة + الطراز (الموديل تحديدًا) + سنة الصنع + المسافة المقطوعة** "
        "مُدخلاتٍ، وقد دُرِّب على لوغاريتم السعر log(Price). واختُبر على **{test:,} سيارة** "
        "(20% من {rows:,}) لم يرها أثناء التدريب:\n\n"
        "- **R² = {r2:.2f}**: يفسّر نحو {r2p} من التباين في الأسعار.\n"
        "- **MAE = {mae:,} ريال سعودي**: أي أن تقديره يبعد في المتوسط بهذا المقدار تقريبًا عن "
        "السعر الفعلي (متوسط الخطأ المطلق). وللمقارنة، يبلغ السعر الوسيط للسيارة في البيانات "
        "59,000 ريال.\n\n"
        "**قيد معروف:** لا يعرف النموذج حالة السيارة، ولا فئة التجهيز (مثل «فل» مقابل «ستاندرد»)، "
        "ولا لونها، ولا سجلّ الحوادث، فهذه المعلومات غير موجودة في البيانات بعد التنظيف. "
        "لذا قد ينحرف التقدير في سيارة مجهّزة تجهيزًا استثنائيًا أو متضرّرة قياسًا بماركتها "
        "وسنة صنعها ومسافتها.",
    ),
    "faq_q2": L("Why use Linear Regression?", "لماذا اختير الانحدار الخطي؟"),
    "faq_a2": L(
        "It was a deliberate choice for simplicity, for three reasons:\n\n"
        "- **It fits the problem.** The goal is to predict a continuous number (a price) "
        "from several factors, which is exactly what Linear Regression does.\n"
        "- **It's easy to explain.** It's simple enough to present to a non-technical "
        "audience.\n"
        "- **It suits the project's scope.** This is a short, beginner-level learning "
        "project.\n\n"
        "To be upfront: **no other model was tried or compared.** Linear Regression wasn't "
        "picked because it beat the alternatives; it was picked because it was the right "
        "size for this project. A different model might well predict more accurately.\n\n"
        "For the record, it was trained on 4,384 cars and tested on 1,096 it had never "
        "seen (an 80/20 split, random_state=103).",
        "كان اختيارًا مقصودًا لصالح البساطة، لثلاثة أسباب:\n\n"
        "- **يناسب المسألة.** الهدف توقّع قيمة رقمية متصلة (السعر) اعتمادًا على عدة عوامل، "
        "وهذا بالضبط ما يقوم به الانحدار الخطي.\n"
        "- **سهل الشرح.** فهو بسيط بما يكفي لعرضه على جمهور غير متخصص.\n"
        "- **يناسب نطاق المشروع.** فهذا مشروع تعليمي قصير بمستوى المبتدئين.\n\n"
        "وللشفافية: **لم يُجرَّب أي نموذج آخر ولم تُجرَ أي مقارنة.** فلم يُختر الانحدار الخطي "
        "لأنه تفوّق على البدائل، بل لأنه الأنسب لحجم هذا المشروع. وقد يقدّم نموذج مختلف "
        "تنبؤات أدق.\n\n"
        "للتوثيق: دُرِّب النموذج على 4,384 سيارة واختُبر على 1,096 سيارة لم يرها من قبل "
        "(تقسيم 80/20، random_state=103).",
    ),
    "faq_q3": L(
        "Why include the car's specific model (Type) and not just the make?",
        "لماذا أُدرج طراز السيارة (Type) وليس الماركة وحدها؟",
    ),
    "faq_a3": L(
        "Because the same brand covers cars worlds apart: a Corolla and a Land Cruiser "
        "are both Toyotas. It was the single biggest improvement in the whole project:\n\n"
        "- With **Make + Year + Mileage** only, the model scored **R² = 0.54**, with an "
        "average error of **28,921 SAR**.\n"
        "- Adding **Type** lifted it to **R² = 0.83** and cut the error to "
        "**14,686 SAR**.\n\n"
        "The trade-off: there are **381 distinct types**, and 107 of them appear for only "
        "one car. Predictions for very rare models carry more uncertainty, because the "
        "model has almost nothing to learn from.",
        "لأن الماركة الواحدة تضم سيارات مختلفة تمامًا: فالكورولا ولاند كروزر كلتاهما من تويوتا. "
        "وكانت هذه أكبر خطوة تحسين في المشروع كله:\n\n"
        "- باستخدام **الماركة + سنة الصنع + المسافة المقطوعة** فقط، سجّل النموذج **R² = 0.54** "
        "بمتوسط خطأ **28,921 ريالًا**.\n"
        "- وبإضافة **الطراز** ارتفع إلى **R² = 0.83** وانخفض الخطأ إلى **14,686 ريالًا**.\n\n"
        "والمقابل: يوجد **381 طرازًا مختلفًا**، منها 107 طرازات لا تظهر إلا في سيارة واحدة. "
        "لذا فالتنبؤات للطرازات النادرة جدًا أقل يقينًا، لأن النموذج لا يجد أمامه إلا القليل "
        "ليتعلّم منه.",
    ),
    "faq_q4": L(
        "Why is the model trained on log(Price) instead of the raw price?",
        "لماذا يُدرَّب النموذج على لوغاريتم السعر بدلًا من السعر الخام؟",
    ),
    "faq_a4": L(
        "Prices in this dataset are heavily **right-skewed**: the average price "
        "(80,157 SAR) is well above the median (59,000 SAR), because 112 cars priced "
        "above 300,000 SAR stretch the tail. On raw prices, that long tail distorts a "
        "straight-line fit.\n\n"
        "Training on **log(Price)** compresses the tail, which improved accuracy. The "
        "model's output is then converted back to a normal price in SAR with `np.exp()`, "
        "so what you see on the Price Estimator tab is always in SAR.",
        "أسعار هذه البيانات **ملتوية نحو اليمين** بشدة: فمتوسط السعر (80,157 ريالًا) أعلى بكثير "
        "من الوسيط (59,000 ريال)، لأن 112 سيارة يزيد سعرها على 300,000 ريال تمدّد الذيل. "
        "وعلى الأسعار الخام يشوّه هذا الذيل الطويل ملاءمة الخط المستقيم.\n\n"
        "أما التدريب على **لوغاريتم السعر** فيضغط هذا الذيل، وقد حسّن الدقة. ثم يُحوَّل ناتج "
        "النموذج إلى سعر عادي بالريال باستخدام `np.exp()`، فما تراه في تبويب مُقدِّر السعر "
        "يكون دائمًا بالريال السعودي.",
    ),
    "faq_q5": L("Why were some listings removed?", "لماذا حُذفت بعض الإعلانات؟"),
    "faq_a5": L(
        "The raw file had **{raw:,} listings**; **{rows:,}** "
        "({kept}) remain after cleaning. Here is exactly what was removed, "
        "in order:\n\n"
        "- **{dup} duplicate rows**: the same listing counted twice.\n"
        "- **{neg:,} listings priced “Negotiable”**: they have no actual "
        "price, so there's nothing for a price model to learn from. This is by far the "
        "biggest cut.\n"
        "- **1 listing with a price of zero.**\n"
        "- **28 cars older than 1990**: too few and too unusual to learn from.\n"
        "- **35 cars with mileage above 600,000 km**: extreme outliers that would "
        "distort the mileage pattern.\n"
        "- **88 modern (2010 or newer) cars priced under 5,000 SAR**: almost certainly "
        "data-entry errors, not real deals.\n"
        "- **399 mileage values corrected, not removed**: they looked mistyped in "
        "thousands (e.g. “300” meaning 300,000 km), so they were multiplied by 1,000. "
        "4 of those cars were still unrealistic after the fix and were dropped.\n\n"
        "That's {removed:,} listings in total: {neg:,} "
        "“Negotiable”, {dup} duplicates, and {imp} rows "
        "with implausible values.",
        "احتوى الملف الخام على **{raw:,} إعلان**؛ وبقي منها **{rows:,}** ({kept}) بعد التنظيف. "
        "وفيما يلي ما حُذف بالضبط، بالترتيب:\n\n"
        "- **{dup} صفًّا مكرّرًا**: الإعلان نفسه محسوب مرتين.\n"
        "- **{neg:,} إعلان سعره «قابل للتفاوض»**: ليس لها سعر فعلي، فلا شيء يتعلّمه نموذج "
        "الأسعار منها. وهذا أكبر اقتطاع بفارق كبير.\n"
        "- **إعلان واحد سعره صفر.**\n"
        "- **28 سيارة أقدم من عام 1990**: قليلة وغير معتادة بما لا يفيد التعلّم منها.\n"
        "- **35 سيارة تجاوزت مسافتها 600,000 كم**: قيم متطرفة مستحيلة كانت ستشوّه نمط "
        "المسافة المقطوعة.\n"
        "- **88 سيارة حديثة (موديل 2010 فما بعد) سعرها أقل من 5,000 ريال**: على الأرجح "
        "أخطاء في إدخال البيانات لا صفقات حقيقية.\n"
        "- **399 قيمة للمسافة المقطوعة صُحِّحت ولم تُحذف**: بدت مكتوبة بالآلاف خطأً "
        "(مثل «300» بمعنى 300,000 كم)، فضُربت في 1,000. وبقيت 4 سيارات منها غير واقعية "
        "بعد التصحيح فحُذفت.\n\n"
        "فيكون المجموع {removed:,} إعلانًا: {neg:,} «قابل للتفاوض»، و{dup} مكرّرة، "
        "و{imp} صفًّا بقيم غير منطقية.",
    ),
    "faq_q6": L("Why was the data cleaned this way?", "لماذا نُظِّفت البيانات بهذه الطريقة؟"),
    "faq_a6": L(
        "The guiding rule was **remove what is clearly wrong, but keep what is merely "
        "unusual**:\n\n"
        "- **Fix, don't delete, when the intent is obvious.** 399 cars had mileage that "
        "looked mistakenly typed in thousands (e.g. 300 meaning 300,000 km). Those were "
        "multiplied by 1,000 instead of being discarded, which saved good listings.\n"
        "- **The 5,000 SAR price floor only applies to 2010+ cars.** A modern car for under "
        "5,000 SAR is implausible, but an old, high-mileage car for that price can be real, "
        "so older cheap cars were left alone. Only 8 cars in the final data are priced "
        "under 1,000 SAR, and they are genuine old, low-value cars.\n"
        "- **Expensive cars stay.** Rolls-Royces and Bentleys are rare, but they are real "
        "listings, so they were kept rather than trimmed away.",
        "كانت القاعدة الحاكمة: **احذف ما هو خاطئ بوضوح، وأبقِ ما هو غير معتاد فحسب**:\n\n"
        "- **أصلِح ولا تحذف حين يكون القصد واضحًا.** كانت لدى 399 سيارة مسافة مقطوعة بدت مكتوبة "
        "بالآلاف خطأً (مثل 300 بمعنى 300,000 كم). فضُربت في 1,000 بدل حذفها، وبذلك أُنقذت "
        "إعلانات سليمة.\n"
        "- **حدّ السعر الأدنى 5,000 ريال ينطبق على موديلات 2010 فما بعد فقط.** فالسيارة الحديثة "
        "بأقل من 5,000 ريال غير معقولة، أما السيارة القديمة كثيرة الاستخدام بهذا السعر فقد "
        "تكون حقيقية، لذا تُركت السيارات القديمة الرخيصة. ولا يوجد في البيانات النهائية سوى "
        "8 سيارات بأقل من 1,000 ريال، وهي سيارات قديمة منخفضة القيمة حقًّا.\n"
        "- **السيارات الغالية تبقى.** رولز رويس وبنتلي نادرتان، لكنهما إعلانان حقيقيان، "
        "فأُبقيتا بدل استبعادهما.",
    ),
    "faq_q7": L(
        "Where does the data come from, and what's in it?",
        "من أين جاءت البيانات، وماذا تحتوي؟",
    ),
    "faq_a7": L(
        "The data is the **Saudi Arabia Used Cars Dataset** on Kaggle, by Turki Bintaleb, "
        "scraped from the Syarah website. The raw file has 15 columns; only 6 were kept: "
        "**Make, Type, Year, Mileage, Region and Price**, because they're the ones that "
        "actually explain price. Columns such as the listing URL or the “Negotiable” flag "
        "were only useful during cleaning.\n\n"
        "- **61 makes** appear in the data; {make} is the most common with "
        "{n:,} listings ({p} of all cars), yet its median "
        "price (65,000 SAR) is far below Mercedes "
        "(149,500 SAR) or Lexus (142,000 SAR). Common doesn't mean expensive.\n"
        "- **2016** is the most common production year, with 880 cars.\n"
        "- The **cheapest cars** are older, high-mileage models kept on purpose. At the "
        "other end, 49 cars are priced above 400,000 SAR: rare luxury makes like "
        "Rolls-Royce and Bentley.",
        "البيانات من **مجموعة بيانات السيارات المستعملة في السعودية** على Kaggle، من إعداد "
        "تركي بن طالب، وهي مأخوذة من موقع سيارة (Syarah). يحتوي الملف الخام على 15 عمودًا، "
        "أُبقي منها 6 فقط: **الماركة، الطراز، سنة الصنع، المسافة المقطوعة، المنطقة، السعر**، "
        "لأنها الأعمدة التي تفسّر السعر فعلًا. أما أعمدة مثل رابط الإعلان أو علامة «قابل للتفاوض» "
        "فكانت مفيدة أثناء التنظيف فقط.\n\n"
        "- تظهر في البيانات **61 ماركة**؛ وأكثرها شيوعًا {make} بـ {n:,} إعلان ({p} من السيارات)، "
        "ومع ذلك فسعرها الوسيط (65,000 ريال) أقل بكثير من مرسيدس (149,500 ريال) أو لكزس "
        "(142,000 ريال). فالشائع لا يعني الغالي.\n"
        "- **2016** هي أكثر سنوات الصنع شيوعًا، بـ 880 سيارة.\n"
        "- **أرخص السيارات** موديلات قديمة كثيرة الاستخدام أُبقيت عمدًا. وعلى الطرف الآخر، "
        "هناك 49 سيارة يزيد سعرها على 400,000 ريال: ماركات فاخرة نادرة مثل رولز رويس وبنتلي.",
    ),
    "faq_q8": L(
        "Does mileage matter on its own, or is it just a stand-in for age?",
        "هل تؤثر المسافة المقطوعة بحد ذاتها، أم أنها مجرد بديل عن العمر؟",
    ),
    "faq_a8": L(
        "It matters on its own. Mileage and Price correlate at **-0.35** (Year and "
        "Mileage correlate at -0.60, confirming older cars do rack up more kilometers). "
        "Grouped by mileage band, the median price falls steadily from **103,000 SAR "
        "under 30,000 km** down to **32,000 SAR above 300,000 km**.\n\n"
        "The key check: even among cars of the same production years (2015–2018 only), "
        "the median price still drops from **85,000 SAR (under 60,000 km)** to "
        "**58,000 SAR (200,000+ km)**. So mileage isn't merely age in disguise. Overall, "
        "94.2% of cars with over 200,000 km are priced at 100,000 SAR or less.",
        "إنها تؤثر بحد ذاتها. فالارتباط بين المسافة المقطوعة والسعر **‎-0.35** (وبين سنة "
        "الصنع والمسافة المقطوعة ‎-0.60، وهذا يؤكد أن السيارات الأقدم تقطع مسافات أكبر فعلًا). "
        "وعند التجميع حسب فئات المسافة ينخفض السعر الوسيط باطّراد من **103,000 ريال لما دون "
        "30,000 كم** إلى **32,000 ريال لما فوق 300,000 كم**.\n\n"
        "والاختبار الحاسم: حتى بين السيارات ذات سنوات الصنع نفسها (2015–2018 فقط)، ينخفض "
        "السعر الوسيط من **85,000 ريال (أقل من 60,000 كم)** إلى **58,000 ريال "
        "(200,000 كم فأكثر)**. فالمسافة المقطوعة ليست مجرد العمر بثوب آخر. وإجمالًا، فإن "
        "94.2% من السيارات التي تجاوزت 200,000 كم سعرها 100,000 ريال أو أقل.",
    ),
    "faq_q9": L("Why isn't Region one of the model's inputs?", "لماذا ليست المنطقة من مُدخلات النموذج؟"),
    "faq_a9": L(
        "Region is a real pattern: among the 8 regions with at least 100 cars (4,960 of "
        "5,480 cars), **Dammam** has the highest median price (**69,000 SAR**) and "
        "**Makkah** the lowest (**43,000 SAR**).\n\n"
        "It was left out of the model to **keep the input list simple** for the prediction "
        "tool (only Make, Type, Year and Mileage). Region also interacts heavily with make "
        "and type, and modelling that reliably would need more data per region than this "
        "dataset has. So it's shown here as an interesting pattern, not as a model input.",
        "المنطقة نمط حقيقي: فمن بين المناطق الثماني التي تضم 100 سيارة على الأقل "
        "(4,960 من أصل 5,480 سيارة)، سجّلت **الدمام** أعلى سعر وسيط (**69,000 ريال**) "
        "وسجّلت **مكة المكرمة** أدناه (**43,000 ريال**).\n\n"
        "لكنها استُبعدت من النموذج **للإبقاء على قائمة المُدخلات بسيطة** في أداة التنبؤ "
        "(الماركة والطراز وسنة الصنع والمسافة المقطوعة فقط). كما أن المنطقة تتفاعل بقوة مع "
        "الماركة والطراز، ونمذجة ذلك بموثوقية تتطلب بيانات لكل منطقة أكثر مما تتيحه هذه "
        "المجموعة. ولذلك تُعرض هنا بوصفها نمطًا لافتًا، لا مُدخلًا في النموذج.",
    ),
    "faq_q10": L(
        "Why use the median price instead of the average?",
        "لماذا نستخدم السعر الوسيط بدل المتوسط؟",
    ),
    "faq_a10": L(
        "Because a few very expensive cars pull the average up. The median price is "
        "**59,000 SAR**: half the cars cost less, half cost more, while the average is "
        "**80,157 SAR**, dragged upward by 112 cars priced above 300,000 SAR.\n\n"
        "The same effect shows up inside a single make: Mercedes' average price "
        "(175,689 SAR) sits well above its median (149,500 SAR). That's why the charts "
        "and the “typical price” comparison in each estimate use medians.",
        "لأن قلّة من السيارات باهظة الثمن ترفع المتوسط. فالسعر الوسيط **59,000 ريال**: "
        "نصف السيارات أرخص منه ونصفها أغلى، بينما المتوسط **80,157 ريالًا**، إذ رفعته "
        "112 سيارة يزيد سعرها على 300,000 ريال.\n\n"
        "ويظهر الأثر نفسه داخل الماركة الواحدة: فمتوسط سعر مرسيدس (175,689 ريالًا) أعلى "
        "بوضوح من وسيطها (149,500 ريال). ولهذا تعتمد الرسوم البيانية ومقارنة «السعر المعتاد» "
        "في كل تقدير على الوسيط.",
    ),
    "faq_q11": L(
        "Why do some charts show only part of the data?",
        "لماذا تعرض بعض الرسوم جزءًا من البيانات فقط؟",
    ),
    "faq_a11": L(
        "- **Make chart:** it shows the most common makes by listing count. Honda and "
        "Mazda tie for 10th place with 142 cars each, so both are included, for 11 makes in "
        "total.\n"
        "- **Year chart:** only production years with at least 20 cars are charted "
        "(2001–2021, covering 5,388 of the 5,480 cars). The 92 older cars from 1990–2000 "
        "are too thin per year to give a reliable median. Year and Price correlate at "
        "0.35 across the full dataset.\n"
        "- **Price chart:** all cars are included; the long right tail is real "
        "(see the log(Price) question above).",
        "- **رسم الماركات:** يعرض أكثر الماركات شيوعًا بحسب عدد الإعلانات. وتتعادل هوندا ومازدا "
        "في المركز العاشر بـ 142 سيارة لكل منهما، فأُدرجتا معًا، أي 11 ماركة إجمالًا.\n"
        "- **رسم سنة الصنع:** لا تُرسم إلا سنوات الصنع التي لا يقل عدد سياراتها عن 20 "
        "(من 2001 إلى 2021، وتغطي 5,388 من أصل 5,480 سيارة). أما السيارات الـ 92 الأقدم "
        "(من 1990 إلى 2000) فقليلة في كل سنة بما لا يكفي لحساب وسيط موثوق. ويبلغ الارتباط "
        "بين سنة الصنع والسعر 0.35 على مستوى البيانات كاملة.\n"
        "- **رسم الأسعار:** جميع السيارات مشمولة؛ والذيل الطويل نحو اليمين حقيقي "
        "(انظر سؤال لوغاريتم السعر أعلاه).",
    ),
}

# The language switcher's own labels (shown the same way in both languages)
LANGUAGE_LABELS = {"en": "EN", "ar": "عربي"}


# ---------------------------------------------------------------
# Language selection. The switcher sits above the tabs, so it shows on every tab.
# session_state["lang"] holds "en" or "ar"; changing it re-runs the script
# instantly, and every string below is looked up again through t().
# ---------------------------------------------------------------
def keep_language_selected():
    """Clicking the already-selected pill would un-select it; put the choice back instead."""
    if st.session_state.get("lang") is None:
        st.session_state["lang"] = st.session_state.get("lang_last", "en")
    st.session_state["lang_last"] = st.session_state["lang"]


st.session_state.setdefault("lang", "en")
with st.container(key="pills_lang"):
    if hasattr(st, "segmented_control"):
        st.segmented_control(
            "Language",
            ["en", "ar"],
            format_func=LANGUAGE_LABELS.get,
            key="lang",
            label_visibility="collapsed",
            on_change=keep_language_selected,
        )
    else:  # older Streamlit versions: same behavior, plainer look
        st.radio(
            "Language",
            ["en", "ar"],
            format_func=LANGUAGE_LABELS.get,
            horizontal=True,
            key="lang",
            label_visibility="collapsed",
        )
LANG = st.session_state.get("lang") or "en"
IS_RTL = LANG == "ar"


def t(key: str, **values) -> str:
    """Look up a piece of text in the current language and fill in its {placeholders}."""
    text = TEXT[key][LANG]
    return text.format(**values) if values else text


def isolate(text: str) -> str:
    """In Arabic sentences, keep Latin names (e.g. "Toyota") from scrambling the punctuation
    next to them, by wrapping them in Unicode bidi isolate marks. English text is unchanged."""
    return f"⁨{text}⁩" if IS_RTL else text


# ---------------------------------------------------------------
# Custom styling (CSS)
# CSS is the language that controls colors, fonts, spacing and borders on a web page.
# The colors are stored once as "variables" (--accent, --text ...) and reused below.
# ---------------------------------------------------------------
CSS_VARIABLES = f"""
:root {{
    --bg: {PAGE_BACKGROUND};
    --accent: {ACCENT};
    --accent-hover: {ACCENT_HOVER};
    --accent-soft: {ACCENT_SOFT};
    --border: {CARD_BORDER};
    --text: {TEXT_COLOR};
    --muted: {MUTED_TEXT};
    --shadow: 0 1px 2px rgba(44, 62, 80, 0.04), 0 6px 20px rgba(110, 193, 228, 0.14);
    --shadow-hover: 0 2px 4px rgba(44, 62, 80, 0.06), 0 14px 30px rgba(110, 193, 228, 0.26);
}}
"""

CSS_STYLES = """
/* ============ Page ============ */
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 4.5rem 2rem 4rem; }

/* ============ Tab bar: physical "folder tab" shapes ============
   Inactive tabs sit slightly lower and behind, in the pale accent color.
   The active tab is raised, brought to the front (z-index) and colored
   white so it visually merges into the "folder body" (the panel) below it. */
[data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: none !important;
    position: relative;
    z-index: 2;
    padding-left: 0.2rem;
}
button[data-baseweb="tab"] {
    height: auto;
    padding: 0.85rem 1.6rem;
    margin-right: -12px;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 14px 14px 0 0;
    background: var(--accent-soft);
    position: relative;
    top: 7px;
    z-index: 1;
    transition: top 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}
button[data-baseweb="tab"]:hover { top: 3px; background: #FFFFFF; }
button[data-baseweb="tab"] p {
    font-size: 1rem;
    font-weight: 500;
    color: var(--muted);
    transition: color 0.15s ease;
}
button[data-baseweb="tab"]:hover p { color: var(--text); }
button[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF;
    top: 0;
    z-index: 3;
    box-shadow: 0 -8px 16px rgba(110, 193, 228, 0.16), var(--shadow);
}
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--text); font-weight: 700; }
/* A thin white sliver so the active tab's bottom edge fuses with the panel beneath it */
button[data-baseweb="tab"][aria-selected="true"]::after {
    content: "";
    position: absolute;
    left: 1px; right: 1px; bottom: -2px;
    height: 3px;
    background: #FFFFFF;
}
[data-baseweb="tab-border"] { display: none; }
/* The active tab is already shown via its own raised, white "folder tab" styling
   above, so BaseWeb's default sliding underline would be a redundant second
   indicator, so hide it. */
[data-baseweb="tab-highlight"] { display: none; }

/* ============ The "folder body": one shared panel holding each tab's content ============ */
div[role="tabpanel"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    box-shadow: var(--shadow);
    padding: 2.4rem 2.6rem 2.8rem;
    position: relative;
    z-index: 2;
    margin-top: -1px;
}

/* ============ Text inside the tabs ============ */
div[role="tabpanel"] p, div[role="tabpanel"] label,
div[role="tabpanel"] [data-testid="stWidgetLabel"] p { color: var(--text); }
div[role="tabpanel"] [data-testid="stWidgetLabel"] p { font-weight: 600; font-size: 0.9rem; }
/* Field labels are drawn by us (so the language can change without resetting the
   input's value); they look just like Streamlit's own widget labels. */
.field-label { font-weight: 600; font-size: 0.9rem; color: var(--text); margin-bottom: 0.3rem; }

/* ---- Headings we draw ourselves (clear size hierarchy) ---- */
.page-header { margin-bottom: 1.75rem; }
.page-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text); line-height: 1.2; }
.page-byline { font-size: 0.95rem; font-weight: 600; color: var(--muted); margin-top: 0.45rem; }
.page-subtitle { font-size: 1.05rem; color: var(--muted); margin-top: 0.4rem; max-width: 760px; }
.section-title { font-size: 1.25rem; font-weight: 700; color: var(--text); margin: 2.5rem 0 0.25rem; }
.section-subtitle { font-size: 0.95rem; color: var(--muted); margin: 0 0 1.1rem; max-width: 760px; }
/* A thin rule that separates the "data" half of the Estimator page from the tool below it */
.section-divider { border-top: 1px solid var(--border); margin: 2.6rem 0 2.2rem; }

/* ============ Hero banner (Predict Price) ============ */
.hero {
    background: linear-gradient(135deg, #FFFFFF 0%, var(--accent-soft) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.4rem 1.9rem;
    box-shadow: var(--shadow);
}
.pill {
    display: inline-block;
    background: rgba(110, 193, 228, 0.22);
    color: var(--text);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.15;
    color: var(--text);
    margin: 0.9rem 0 0.7rem;
    max-width: 780px;
}
.hero-lead { font-size: 1.05rem; line-height: 1.6; color: var(--muted); max-width: 720px; margin: 0; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1.1rem; }
.chip {
    background: #FFFFFF;
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 0.85rem;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
}
.chip b { font-weight: 700; }

/* ============ Folder-tab flourish shared by EVERY content card ============
   A small pale-blue tab shape peeks out of the top-left of each card, echoing
   the tab bar above and reinforcing the "document in a folder" feel. Reused,
   unchanged, on stat cards, trust cards, driver cards and chart cards. */
.stat-card, .trust-card, .driver-card, [class*="st-key-card_"] {
    position: relative;
    overflow: visible;
    margin-top: 0.9rem;
}
.stat-card::after, .trust-card::after, .driver-card::after, [class*="st-key-card_"]::after {
    content: "";
    position: absolute;
    top: -9px;
    left: 22px;
    width: 46px;
    height: 12px;
    background: var(--accent-soft);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 7px 7px 0 0;
}

/* ============ Stat cards ("Data at a glance") ============ */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; }
.stat-card, .trust-card, .driver-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.35rem 1.7rem 1.3rem;
    box-shadow: var(--shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stat-card::before, .trust-card::before, .driver-card::before {
    content: "";
    position: absolute;
    left: 0; top: 1.3rem; bottom: 1.3rem;
    width: 4px;
    border-radius: 0 4px 4px 0;
    background: var(--accent);
}
.stat-card:hover, .trust-card:hover, .driver-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); }
.stat-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
}
.stat-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text); margin: 0.3rem 0 0.15rem; line-height: 1.15; }
.stat-note { font-size: 0.88rem; color: var(--muted); }

/* ============ Trust cards: "Data quality" and "Model confidence" ============ */
.trust-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 0.6rem; }
.trust-headline { font-size: 1.15rem; font-weight: 700; color: var(--text); margin: 0.4rem 0 0.2rem; line-height: 1.35; }
.trust-headline b { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; }
.trust-text { font-size: 0.9rem; line-height: 1.55; color: var(--muted); margin: 0.7rem 0 0; }
.trust-metrics { display: flex; flex-wrap: wrap; gap: 0.5rem 2.4rem; margin-top: 0.35rem; }
.trust-metric-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text); line-height: 1.15; }
.trust-metric-value span { font-size: 1rem; font-weight: 600; color: var(--muted); letter-spacing: 0; }
.trust-metric-label { font-size: 0.85rem; color: var(--muted); }
.trust-legend { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); margin-top: 0.45rem; }
.trust-legend b { color: var(--text); }

/* ============ Cards: any container whose key starts with "card_" ============ */
[class*="st-key-card_"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.6rem 1.4rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.15s ease;
}
[class*="st-key-card_"]:hover { box-shadow: var(--shadow-hover); }
/* Chart cards on the Analysis tab: one per row, roomier inside and well spaced apart */
[class*="st-key-card_chart_"] { padding: 1.9rem 2.1rem 1.9rem; margin-bottom: 1.6rem; }
.card-title {
    font-size: 1.1rem; font-weight: 700; color: var(--text);
    border-left: 4px solid var(--accent); padding-left: 0.7rem; margin-bottom: 1rem;
}

/* ============ Insight callout under each chart ============
   A soft blue gradient, a thick accent border on the left, a round badge with a
   lightbulb icon, and a small "Key takeaway" label above larger body text. */
.insight {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-top: 1.2rem;
    padding: 1.1rem 1.4rem 1.15rem 1.2rem;
    background: linear-gradient(90deg, var(--accent-soft) 0%, #FBFDFF 100%);
    border-left: 5px solid var(--accent);
    border-radius: 4px 14px 14px 4px;
}
.insight-icon {
    flex: none;
    width: 36px; height: 36px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--text);
    display: flex; align-items: center; justify-content: center;
}
.insight-icon svg { width: 20px; height: 20px; }
.insight-label {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.2rem;
}
.insight-text { font-size: 1.02rem; line-height: 1.6; color: var(--text); margin: 0; }

/* ============ Driver cards ("What drives the price?") ============
   A 6-column grid: the first row holds three cards (2 columns each) and the
   second row two wider cards (3 columns each). No connecting lines, just cards. */
.driver-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1.4rem 1.25rem; }
.driver-card { grid-column: span 2; padding: 1.4rem 1.7rem 1.5rem; }
.driver-card:nth-child(n+4) { grid-column: span 3; }
.driver-label {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--muted);
}
.driver-title { font-size: 1.15rem; font-weight: 700; color: var(--text); margin: 0.35rem 0 0.7rem; line-height: 1.3; }
.driver-figure {
    display: inline-block;
    background: var(--accent-soft);
    color: var(--text);
    font-size: 0.85rem; font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
}
.driver-note { font-size: 0.92rem; line-height: 1.55; color: var(--muted); margin: 0.75rem 0 0; }
.driver-tag {
    display: inline-block; margin-left: 0.5rem;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 0.15rem 0.55rem;
    vertical-align: middle;
}

/* ============ FAQ: st.expander styled as a website FAQ ============
   Each question is its own independent expander (collapsed by default), so any
   number can be open or closed at once. */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: var(--shadow);
    margin-bottom: 0.7rem;
    overflow: hidden;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
[data-testid="stExpander"] details { border: none !important; border-radius: 14px; }
[data-testid="stExpander"] summary { padding: 1rem 1.3rem; transition: background 0.15s ease; }
[data-testid="stExpander"] summary:hover { background: var(--accent-soft); }
[data-testid="stExpander"] summary p { font-size: 1.02rem; font-weight: 600; color: var(--text); }
[data-testid="stExpander"] summary svg { color: var(--muted); }
[data-testid="stExpander"]:has(details[open]) {
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-hover);
}
[data-testid="stExpander"] details[open] summary { background: var(--accent-soft); }
[data-testid="stExpanderDetails"] { padding: 0.9rem 1.4rem 1.3rem; }
[data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] li {
    font-size: 0.97rem; line-height: 1.65; color: var(--text);
}
[data-testid="stExpanderDetails"] li { margin-bottom: 0.3rem; }

/* ============ Pill selectors: the language switcher and the FAQ filter ============
   st.segmented_control draws one joined bar by default; here each option becomes its own
   rounded pill, echoing the "Try it yourself" badge. The active pill is filled sky blue. */
[class*="st-key-pills_"] { margin-bottom: 1.1rem; }
[class*="st-key-pills_"] [data-testid="stButtonGroup"] { gap: 0.55rem; flex-wrap: wrap; }
[class*="st-key-pills_"] [data-testid^="stBaseButton-segmented_control"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 999px !important;
    padding: 0.4rem 1.1rem;
    min-height: 0;
    box-shadow: none;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[class*="st-key-pills_"] [data-testid^="stBaseButton-segmented_control"] p {
    color: var(--text);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[class*="st-key-pills_"] [data-testid^="stBaseButton-segmented_control"]:hover {
    background: var(--accent-soft);
    border-color: var(--accent);
}
[class*="st-key-pills_"] [data-testid="stBaseButton-segmented_controlActive"] {
    background: var(--accent);
    border-color: var(--accent);
}
/* The language switcher sits at the end of the row, above the tabs */
.st-key-pills_lang { align-items: flex-end; margin-bottom: 0.9rem; }
.st-key-pills_lang [data-testid^="stBaseButton-segmented_control"] p { letter-spacing: 0; text-transform: none; font-size: 0.9rem; }
.faq-count { font-size: 0.85rem; color: var(--muted); margin: 0 0 0.8rem 0.2rem; }
/* "About the data": a quiet attribution note above the FAQ filter, not a headline */
.about-data { margin: 0 0 1.3rem; padding: 0.2rem 0 0.2rem 1rem; border-left: 3px solid var(--border); max-width: 780px; }
.about-data-title { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
div[role="tabpanel"] .about-data-text { font-size: 0.92rem; line-height: 1.6; color: var(--muted); margin: 0.25rem 0 0; }

/* ============ Inputs: pale blue borders, sky blue when focused (no red) ============ */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {
    border-color: var(--border) !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}
[data-baseweb="select"]:focus-within > div, [data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
    background: var(--bg);
    color: var(--text);
}

/* ============ The "Predict Price" button ============ */
.st-key-predict_button, .st-key-predict_button [data-testid="stButton"] { width: 100%; }
.st-key-predict_button button {
    width: 100%;
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: var(--text) !important;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.65rem 1.8rem;
    margin-top: 0.4rem;
    transition: background 0.15s ease, box-shadow 0.15s ease;
}
.st-key-predict_button button p { color: var(--text) !important; font-size: 1rem; }
.st-key-predict_button button:hover,
.st-key-predict_button button:focus:not(:active) {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    box-shadow: 0 0 0 0.2rem rgba(110, 193, 228, 0.35) !important;
}

/* ============ Warning box recolored to blue ============ */
[data-testid="stAlert"], [data-testid="stAlert"] > div, [data-testid="stAlertContainer"] {
    background: var(--accent-soft) !important;
    border-radius: 12px;
}
[data-testid="stAlert"] { border: 1px solid var(--border); }
[data-testid="stAlert"] p { color: var(--text); }

/* ============ Prediction panel ============ */
.placeholder-card {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    background: rgba(234, 246, 252, 0.4);
}
.placeholder-title { font-size: 1.15rem; font-weight: 700; color: var(--text); }
.placeholder-text { color: var(--muted); margin: 0.4rem 0 0; }

.result-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 16px;
    padding: 1.8rem 1.9rem 1.7rem;
    box-shadow: var(--shadow-hover);
}
.result-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
}
.result-price { font-size: 3.4rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; color: var(--text); margin: 0.4rem 0 0.9rem; }
.result-price span { font-size: 1.3rem; font-weight: 600; letter-spacing: 0; color: var(--muted); }
.result-section { border-top: 1px solid var(--border); margin-top: 1.3rem; padding-top: 1.1rem; }
.result-heading { font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
.result-text { color: var(--text); line-height: 1.6; margin: 0; }
.meter { height: 8px; background: var(--accent-soft); border-radius: 999px; margin-top: 0.8rem; overflow: hidden; }
.meter > span { display: block; height: 100%; background: var(--accent); border-radius: 999px; }

/* ============ Small screens ============ */
@media (max-width: 900px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .trust-grid { grid-template-columns: 1fr; }
    .driver-grid { grid-template-columns: repeat(2, 1fr); }
    .driver-card, .driver-card:nth-child(n+4) { grid-column: span 1; }
    .hero-title { font-size: 1.7rem; }
    div[role="tabpanel"] { padding: 1.8rem 1.3rem 2.2rem; }
    [class*="st-key-card_chart_"] { padding: 1.4rem 1.2rem; }
}
@media (max-width: 560px) {
    .stat-grid, .driver-grid { grid-template-columns: 1fr; }
}
"""

# Extra rules that are added ONLY when Arabic is selected. Setting direction: rtl on
# the page flips flex rows, grids, columns and tab order by itself; the rules below
# mirror the things that were positioned with explicit "left"/"right" values, keep the
# charts' own axes left-to-right, and switch off letter-spacing (extra spacing between
# letters breaks the joined shapes of Arabic script).
RTL_STYLES = """
.stApp, [data-testid="stMainBlockContainer"] { direction: rtl; text-align: right; }
.stApp * { letter-spacing: 0 !important; }
[data-testid="stPlotlyChart"], .js-plotly-plot { direction: ltr; }

button[data-baseweb="tab"] { margin-right: 0; margin-left: -12px; }
[data-baseweb="tab-list"] { padding-left: 0; padding-right: 0.2rem; }
div[role="tabpanel"] { border-radius: 18px 4px 18px 18px; }

.stat-card::after, .trust-card::after, .driver-card::after, [class*="st-key-card_"]::after {
    left: auto; right: 22px;
}
.stat-card::before, .trust-card::before, .driver-card::before {
    left: auto; right: 0; border-radius: 4px 0 0 4px;
}
.card-title { border-left: none; padding-left: 0; border-right: 4px solid var(--accent); padding-right: 0.7rem; }
.insight {
    border-left: none;
    border-right: 5px solid var(--accent);
    border-radius: 14px 4px 4px 14px;
    padding: 1.1rem 1.2rem 1.15rem 1.4rem;
    background: linear-gradient(270deg, var(--accent-soft) 0%, #FBFDFF 100%);
}
.driver-tag { margin-left: 0; margin-right: 0.5rem; }
.faq-count { margin: 0 0.2rem 0.8rem 0; }
.about-data { padding: 0.2rem 1rem 0.2rem 0; border-left: none; border-right: 3px solid var(--border); }
.section-title, .section-subtitle, .page-title, .page-byline, .page-subtitle { text-align: right; }
.placeholder-card { text-align: center; }
[data-testid="stExpanderDetails"] ul, [data-testid="stExpanderDetails"] ol { padding-right: 1.4rem; padding-left: 0; }
"""


# ---------------------------------------------------------------
# Loading data and model
# @st.cache_data / @st.cache_resource keep the loaded objects in memory,
# so the files are not re-read every time the user clicks something.
# ---------------------------------------------------------------
@st.cache_data
def load_car_data() -> pd.DataFrame:
    """Read the cleaned used-cars CSV file into a DataFrame."""
    return pd.read_csv("used_cars_clean_v2.csv")


@st.cache_resource
def load_model_and_columns():
    """Load the trained Linear Regression model and its expected column order."""
    trained_model = joblib.load("car_price_model.pkl")
    model_columns = list(joblib.load("model_columns.pkl"))
    return trained_model, model_columns


car_data = load_car_data()
trained_model, model_columns = load_model_and_columns()

# Apply the custom styling to the whole page (plus the RTL rules when Arabic is on)
st.markdown(
    f"<style>{CSS_VARIABLES}{CSS_STYLES}{RTL_STYLES if IS_RTL else ''}</style>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Small helper functions used to build the page
# ---------------------------------------------------------------
def to_html(text: str) -> str:
    """Remove line breaks and indentation so Streamlit treats the text as plain HTML."""
    return " ".join(line.strip() for line in text.splitlines())


def show_html(text: str):
    """Display a block of HTML on the page."""
    st.markdown(to_html(text), unsafe_allow_html=True)


def page_header(title: str, subtitle: str, byline: str = ""):
    """A large title, an optional plain-text byline, and one short "why this matters" line."""
    byline_html = f'<div class="page-byline">{escape(byline)}</div>' if byline else ""
    show_html(
        f"""
        <div class="page-header">
            <div class="page-title">{escape(title)}</div>
            {byline_html}
            <div class="page-subtitle">{escape(subtitle)}</div>
        </div>
        """
    )


def section_header(title: str, subtitle: str):
    """A medium-sized heading that starts a new section of a page."""
    show_html(
        f"""
        <div class="section-title">{escape(title)}</div>
        <div class="section-subtitle">{escape(subtitle)}</div>
        """
    )


def card(name: str):
    """A card container. The key makes the CSS above ("card_...") style it."""
    return st.container(key=f"card_{name}")


def field_label(text: str):
    """The small label above an input (see the .field-label CSS)."""
    show_html(f'<div class="field-label">{escape(text)}</div>')


def style_chart(figure, height: int = 400):
    """Apply the shared look (colors, fonts, spacing) to a Plotly chart. The chart's own
    axes stay left-to-right in Arabic too; only the text around the chart is mirrored."""
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor=ACCENT, font_color=TEXT_COLOR),
    )
    figure.update_xaxes(gridcolor=GRID_COLOR, linecolor=CARD_BORDER, zeroline=False)
    figure.update_yaxes(gridcolor=GRID_COLOR, linecolor=CARD_BORDER, zeroline=False)
    return figure


# A lightbulb icon (inline SVG) for the insight callout
LIGHTBULB_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/>'
    '<path d="M9 18h6"/><path d="M10 22h4"/></svg>'
)


def show_chart_card(name: str, title: str, figure, takeaway: str):
    """Show one chart inside its own full-width card: the title, the chart, then its
    takeaway directly underneath in a highlighted callout. Pairing the chart with its
    insight in the same card is the point: the conclusion never floats far from its
    evidence. (The longer "why" behind each chart lives in the FAQ.)
    """
    with card(f"chart_{name}"):
        show_html(f'<div class="card-title">{escape(title)}</div>')
        st.plotly_chart(figure, config={"displayModeBar": False})
        show_html(
            f"""
            <div class="insight">
                <div class="insight-icon">{LIGHTBULB_ICON}</div>
                <div>
                    <div class="insight-label">{escape(t("insight_label"))}</div>
                    <p class="insight-text">{escape(takeaway)}</p>
                </div>
            </div>
            """
        )


def describe_prediction(make, car_type, year, mileage, predicted_price) -> str:
    """Explain, in a natural and engaging voice, why the estimate came out this way.

    The entered year and mileage are compared with the typical (median) values of the
    same make in the data, so the text changes with the user's inputs. The logic (which
    factor pushes the price up or down) is shared by both languages; the sentences
    themselves are separate templates in TEXT, written natively in each language
    rather than translated word for word.
    """
    same_make_cars = car_data[car_data["Make"] == make]
    typical_price = same_make_cars["Price"].median()
    typical_year = int(round(same_make_cars["Year"].median()))
    typical_mileage = same_make_cars["Mileage"].median()

    # 1) How does the estimate compare with the typical price of this make?
    price_ratio = predicted_price / typical_price
    if price_ratio > 1.10:
        price_direction, opening_key = 1, "desc_open_up"
    elif price_ratio < 0.90:
        price_direction, opening_key = -1, "desc_open_dn"
    else:
        price_direction, opening_key = 0, "desc_open_mid"

    # 2) Is the car newer/older and lower/higher mileage than the typical one?
    #    +1 means "pushes the price up", -1 means "pushes the price down".
    year_effect = 0
    if year - typical_year >= 2:
        year_effect = 1
    elif year - typical_year <= -2:
        year_effect = -1

    mileage_effect = 0
    if mileage < typical_mileage * 0.8:
        mileage_effect = 1
    elif mileage > typical_mileage * 1.25:
        mileage_effect = -1

    # 3) Pick the sentence for this combination (u = pushes up, m = neutral, d = pushes down)
    effect_letter = {1: "u", 0: "m", -1: "d"}
    details_key = f"desc_{effect_letter[year_effect]}_{effect_letter[mileage_effect]}"

    words = dict(
        make=isolate(make),
        type=isolate(car_type),
        year=year,
        km=f"{mileage:,.0f}",
        tk=f"{typical_mileage:,.0f}",
        ty=typical_year,
        p=f"{predicted_price:,.0f}",
        tp=f"{typical_price:,.0f}",
    )
    opening = t(opening_key, **words)
    details = t(details_key, **words)

    # 4) If year/mileage alone don't explain the price gap, say so: the model (type)
    # is doing the rest of the work. Skip this when the "both near typical" sentence
    # above has already pointed at the trim.
    net_effect = year_effect + mileage_effect
    net_direction = (net_effect > 0) - (net_effect < 0)
    if not (year_effect == 0 and mileage_effect == 0) and net_direction != price_direction:
        details += t("desc_extra", **words)

    # 5) The disclaimer, written as a natural closing thought rather than an afterthought
    return f"{opening} {details}. {t('desc_disclaimer')}"


# Create the two tabs: the estimator (with the data behind it), then the patterns behind it.
# The tab labels change with the language, and Streamlit would treat that as new tabs and jump
# back to the first one. So we work out which tab was open (from its label in either language)
# and re-open that same tab, in the new language, through the "default" argument.
def selected_tab_label() -> str:
    """The label (in the current language) of the tab the user had open."""
    previous = st.session_state.get("main_tabs")
    for tab_id in ("estimator", "analysis"):
        if previous in (TEXT[f"tab_{tab_id}"]["en"], TEXT[f"tab_{tab_id}"]["ar"]):
            return t(f"tab_{tab_id}")
    return t("tab_estimator")


predict_tab, analysis_tab = st.tabs(
    [t("tab_estimator"), t("tab_analysis")],
    default=selected_tab_label(),
    key="main_tabs",
    on_change="rerun",
)


# ---------------------------------------------------------------
# TAB 1: Price Estimator: the data at a glance first, then the estimator itself
# ---------------------------------------------------------------
with predict_tab:
    # --- Data at a glance: brief context before the tool ---
    # The project's own byline (the raw data's source is credited under Methodology instead).
    # The name stays in Latin letters in Arabic too, wrapped so it doesn't scramble the colon.
    page_header(
        t("glance_title"),
        t("glance_sub"),
        byline=t("byline", name=isolate("Elaf Alyoubi")),
    )

    number_of_rows, number_of_columns = car_data.shape
    most_common_make = car_data["Make"].value_counts().index[0]
    most_common_make_count = car_data["Make"].value_counts().iloc[0]
    most_common_make_share = f"{most_common_make_count / number_of_rows:.0%}"
    most_common_year = int(car_data["Year"].mode().iloc[0])
    most_common_year_count = int((car_data["Year"] == most_common_year).sum())
    column_names = ("، " if IS_RTL else ", ").join(
        t(f"col_{name}") if f"col_{name}" in TEXT else name for name in car_data.columns
    )

    # Each stat card is a (label, big value, note) triple. The longer explanations
    # behind these numbers live in the FAQ on the Analysis tab.
    stat_cards = [
        (t("stat_rows_label"), f"{number_of_rows:,}", t("stat_rows_note")),
        (t("stat_cols_label"), f"{number_of_columns}", column_names),
        (t("stat_median_label"), f"{car_data['Price'].median():,.0f} {t('currency')}", t("stat_median_note")),
        (
            t("stat_make_label"),
            escape(str(most_common_make)),
            t("stat_make_note", n=most_common_make_count, p=most_common_make_share),
        ),
        (t("stat_year_label"), f"{most_common_year}", t("stat_year_note", n=most_common_year_count)),
        (
            t("stat_range_label"),
            f"{car_data['Price'].min():,.0f} – {car_data['Price'].max():,.0f}",
            t("stat_range_note"),
        ),
    ]
    stat_cards_html = "".join(
        f'<div class="stat-card"><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{value}</div><div class="stat-note">{escape(note)}</div></div>'
        for label, value, note in stat_cards
    )
    st.markdown(f'<div class="stat-grid">{stat_cards_html}</div>', unsafe_allow_html=True)

    # --- Trust row: how clean is the data, and how far can the model be trusted? ---
    removed_total = RAW_LISTINGS - number_of_rows
    kept_share = number_of_rows / RAW_LISTINGS
    show_html(
        f"""
        <div class="trust-grid">
            <div class="trust-card">
                <div class="stat-label">{escape(t("dq_label"))}</div>
                <div class="trust-headline">{t("dq_headline", raw=RAW_LISTINGS, rows=number_of_rows)}</div>
                <div class="meter"><span style="width: {kept_share:.0%};"></span></div>
                <div class="trust-legend">
                    <span>{t("dq_kept", p=f"{kept_share:.0%}")}</span>
                    <span>{t("dq_removed", n=removed_total)}</span>
                </div>
                <p class="trust-text">{escape(t("dq_text", neg=REMOVED_NEGOTIABLE, dup=REMOVED_DUPLICATES, imp=REMOVED_IMPLAUSIBLE))}</p>
            </div>
            <div class="trust-card">
                <div class="stat-label">{escape(t("mc_label"))}</div>
                <div class="trust-metrics">
                    <div>
                        <div class="trust-metric-value">{MODEL_R2:.2f}</div>
                        <div class="trust-metric-label">{escape(t("mc_r2"))}</div>
                    </div>
                    <div>
                        <div class="trust-metric-value">±{MODEL_TYPICAL_ERROR_SAR:,} <span>{escape(t("currency"))}</span></div>
                        <div class="trust-metric-label">{escape(t("mc_mae"))}</div>
                    </div>
                </div>
                <div class="meter"><span style="width: {MODEL_R2:.0%};"></span></div>
                <p class="trust-text">{escape(t("mc_text", n=TEST_SET_SIZE))}</p>
            </div>
        </div>
        """
    )

    show_html('<div class="section-divider"></div>')

    # --- The estimator ---
    show_html(
        f"""
        <div class="hero">
            <span class="pill">{escape(t("hero_pill"))}</span>
            <div class="hero-title">{escape(t("hero_title"))}</div>
            <p class="hero-lead">{escape(t("hero_lead", n=len(car_data)))}</p>
        </div>
        """
    )
    st.write("")

    input_panel, result_panel = st.columns([5, 6], gap="large")

    # NOTE: we do NOT use st.form here. Inside a form, changing the Make would not
    # refresh the Type list until the button is pressed. Without a form, the page
    # re-runs on every change, so the Type dropdown updates right away.
    #
    # The widgets have a fixed (English) label that is hidden, and the translated label
    # is drawn above them by field_label(). Streamlit treats a changed widget label as a
    # brand-new widget, which would reset the user's choices whenever the language changed.
    with input_panel:
        with card("inputs"):
            show_html(f'<div class="card-title">{escape(t("inputs_title"))}</div>')

            make_column, type_column = st.columns(2)

            # Dropdown 1: Make (sorted alphabetically, starting on the most common make)
            available_makes = sorted(car_data["Make"].unique())
            with make_column:
                field_label(t("f_make"))
                selected_make = st.selectbox(
                    "Make", available_makes, index=available_makes.index("Toyota"),
                    key="in_make", label_visibility="collapsed",
                )

            # Dropdown 2: Type, only the models that exist for the selected make
            available_types = sorted(
                car_data.loc[car_data["Make"] == selected_make, "Type"].unique()
            )
            with type_column:
                field_label(t("f_type"))
                selected_type = st.selectbox(
                    "Type / Model", available_types,
                    key=f"in_type_{selected_make}", label_visibility="collapsed",
                )

            # Numeric inputs: limits are taken from the data so values stay realistic
            year_column, mileage_column = st.columns(2)
            with year_column:
                field_label(t("f_year"))
                entered_year = st.number_input(
                    "Production year",
                    min_value=int(car_data["Year"].min()),
                    max_value=int(car_data["Year"].max()),
                    value=2018,
                    step=1,
                    key="in_year",
                    label_visibility="collapsed",
                )
            with mileage_column:
                field_label(t("f_mileage"))
                entered_mileage = st.number_input(
                    "Mileage (km)",
                    min_value=0,
                    max_value=1_000_000,
                    value=100_000,
                    step=1_000,
                    key="in_mileage",
                    label_visibility="collapsed",
                )

            if st.button(t("btn_predict"), key="predict_button"):
                # Remember which car was estimated, so the result stays on screen if the
                # user switches language (which re-runs the script and resets the button).
                st.session_state["estimated_car"] = (
                    selected_make, selected_type, int(entered_year), entered_mileage
                )

    # The result is shown while the inputs still match the car that was estimated
    estimate_is_current = st.session_state.get("estimated_car") == (
        selected_make, selected_type, int(entered_year), entered_mileage
    )

    with result_panel:
        if not estimate_is_current:
            # Nothing predicted yet: show an empty placeholder so the layout stays balanced
            show_html(
                f"""
                <div class="placeholder-card">
                    <div class="placeholder-title">{escape(t("ph_title"))}</div>
                    <p class="placeholder-text">{escape(t("ph_text"))}</p>
                </div>
                """
            )
        else:
            # Check that this Make + Type combination really exists in the data
            combination_exists = (
                (car_data["Make"] == selected_make) & (car_data["Type"] == selected_type)
            ).any()

            if not combination_exists:
                st.warning(t("warn_missing", make=isolate(selected_make), type=isolate(selected_type)))
            else:
                # Step 1: a one-row table with every column the model expects, all zeros,
                # in the exact same order as model_columns.pkl
                new_car_row = pd.DataFrame(
                    np.zeros((1, len(model_columns))), columns=model_columns
                )

                # Step 2: fill in the numeric inputs
                new_car_row["Year"] = entered_year
                new_car_row["Mileage"] = entered_mileage

                # Step 3: switch on (set to 1) the one-hot columns of the chosen make and type
                new_car_row[f"Make_{selected_make}"] = 1
                new_car_row[f"Type_{selected_type}"] = 1

                # Step 4: predict. The model was trained on log(Price), so the raw output
                # is a log value, so np.exp() converts it back to a price in SAR.
                predicted_log_price = trained_model.predict(new_car_row)[0]
                predicted_price = np.exp(predicted_log_price)

                # Step 5: write the plain-language explanation for this exact car
                explanation_text = describe_prediction(
                    selected_make, selected_type, int(entered_year), entered_mileage, predicted_price
                )

                # Step 6: show everything in one result card
                car_name = escape(f"{int(entered_year)} {selected_make} {selected_type}")
                show_html(
                    f"""
                    <div class="result-card">
                        <div class="result-label">{escape(t("res_label"))}</div>
                        <div class="chip-row" style="margin-top: 0.7rem;">
                            <span class="chip"><b><bdi>{car_name}</bdi></b></span>
                            <span class="chip">{entered_mileage:,.0f} {escape(t("km"))}</span>
                        </div>
                        <div class="result-price">{predicted_price:,.0f} <span>{escape(t("currency"))}</span></div>
                        <div class="result-section">
                            <div class="result-heading">{escape(t("res_reliable_h"))}</div>
                            <p class="result-text">{escape(t("res_reliable_t", r2p=f"{MODEL_R2:.0%}", r2=MODEL_R2, mae=MODEL_TYPICAL_ERROR_SAR))}</p>
                            <div class="meter"><span style="width: {MODEL_R2:.0%};"></span></div>
                        </div>
                        <div class="result-section">
                            <div class="result-heading">{escape(t("res_why_h"))}</div>
                            <p class="result-text">{escape(explanation_text)}</p>
                        </div>
                    </div>
                    """
                )


# ---------------------------------------------------------------
# TAB 2: Analysis: each chart paired with its takeaway, a card summary of what
# drives the price, and the methodology accordion explaining the reasoning.
# ---------------------------------------------------------------
with analysis_tab:
    page_header(t("an_title"), t("an_sub"))

    # --- Chart 1: histogram of prices ---
    share_under_100k = (car_data["Price"] < 100_000).mean()
    price_histogram = px.histogram(
        car_data,
        x="Price",
        nbins=50,
        color_discrete_sequence=[ACCENT],
    )
    price_histogram.update_traces(marker_line_color="white", marker_line_width=1)
    price_histogram.update_layout(
        yaxis_title=t("ch_price_y"), xaxis_title=t("ch_price_x"), bargap=0.02
    )
    price_histogram.update_xaxes(tickformat=",")
    show_chart_card(
        "price_distribution",
        t("ch_price_title"),
        style_chart(price_histogram),
        t("ch_price_take", p=f"{share_under_100k:.0%}"),
    )

    # --- Chart 2: median price of the 10 most common makes ---
    top_10_makes = car_data["Make"].value_counts().head(10).index
    median_price_by_make = (
        car_data[car_data["Make"].isin(top_10_makes)]
        .groupby("Make")["Price"]
        .median()
        .sort_values()          # sorted so the most expensive make ends up on top
        .reset_index()
    )
    make_bar_chart = px.bar(
        median_price_by_make,
        x="Price",
        y="Make",
        orientation="h",        # "h" = horizontal bars
        color_discrete_sequence=[ACCENT],
        text="Price",
    )
    make_bar_chart.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,.0f} " + t("currency") + "<extra></extra>",
    )
    make_bar_chart.update_layout(xaxis_title=t("ch_make_x"), yaxis_title=None, bargap=0.3)
    make_bar_chart.update_xaxes(tickformat=",", range=[0, median_price_by_make["Price"].max() * 1.15])
    priciest_make = median_price_by_make.iloc[-1]
    cheapest_make = median_price_by_make.iloc[0]
    show_chart_card(
        "median_by_make",
        t("ch_make_title"),
        style_chart(make_bar_chart, height=440),
        t(
            "ch_make_take",
            top=isolate(priciest_make["Make"]),
            low=isolate(cheapest_make["Make"]),
            ratio=priciest_make["Price"] / cheapest_make["Price"],
        ),
    )

    # --- Chart 3: median price by production year ---
    median_price_by_year = car_data.groupby("Year")["Price"].median().reset_index()
    year_line_chart = px.line(
        median_price_by_year,
        x="Year",
        y="Price",
        markers=True,
        color_discrete_sequence=[ACCENT],
    )
    year_line_chart.update_traces(
        line_width=3,
        marker=dict(size=7, line=dict(color="white", width=2)),
        hovertemplate="%{x}: %{y:,.0f} " + t("currency") + "<extra></extra>",
    )
    year_line_chart.update_layout(xaxis_title=t("ch_year_x"), yaxis_title=t("ch_year_y"))
    year_line_chart.update_yaxes(tickformat=",")
    median_price_2010 = median_price_by_year.loc[median_price_by_year["Year"] == 2010, "Price"].iloc[0]
    median_price_2020 = median_price_by_year.loc[median_price_by_year["Year"] == 2020, "Price"].iloc[0]
    show_chart_card(
        "median_by_year",
        t("ch_year_title"),
        style_chart(year_line_chart),
        t("ch_year_take", a=median_price_2010, b=median_price_2020),
    )

    # --- Chart 4: price vs. mileage ---
    mileage_scatter = px.scatter(
        car_data,
        x="Mileage",
        y="Price",
        opacity=0.4,            # semi-transparent dots so overlapping points are visible
        color_discrete_sequence=[ACCENT],
    )
    mileage_scatter.update_traces(
        marker=dict(size=5),
        hovertemplate="%{x:,.0f} " + t("km") + ": %{y:,.0f} " + t("currency") + "<extra></extra>",
    )
    mileage_scatter.update_layout(xaxis_title=t("ch_km_x"), yaxis_title=t("ch_km_y"))
    mileage_scatter.update_xaxes(tickformat=",")
    mileage_scatter.update_yaxes(tickformat=",")
    show_chart_card(
        "price_vs_mileage",
        t("ch_km_title"),
        style_chart(mileage_scatter),
        t("ch_km_take"),
    )

    # --- What drives the price: five plain cards, no diagram ---
    section_header(t("drv_title"), t("drv_sub"))

    driver_ids = ["make", "year", "mileage", "type", "region"]
    driver_cards_html = "".join(
        f'<div class="driver-card"><div class="driver-label">{escape(t(f"drv_{d}_label"))}'
        + (f'<span class="driver-tag">{escape(t("drv_region_tag"))}</span>' if d == "region" else "")
        + f'</div><div class="driver-title">{escape(t(f"drv_{d}_title"))}</div>'
        f'<span class="driver-figure">{escape(t(f"drv_{d}_figure"))}</span>'
        f'<p class="driver-note">{escape(t(f"drv_{d}_note"))}</p></div>'
        for d in driver_ids
    )
    st.markdown(f'<div class="driver-grid">{driver_cards_html}</div>', unsafe_allow_html=True)

    # --- Methodology & Key Decisions: an accordion of independent, collapsed-by-default
    # expanders, ordered from most to least important for trusting the model:
    #   1) the model itself (accuracy, why Linear Regression, why Type, why log(Price))
    #   2) the data behind it (removals, cleaning rationale, source)
    #   3) supporting details (mileage vs. age, Region, median vs. mean, chart filters)
    section_header(t("faq_title"), t("faq_sub"))

    # A short, low-key attribution note: the source of the raw data, stated as one
    # factual detail among the methodology notes (not as front-page branding).
    show_html(
        f"""
        <div class="about-data">
            <div class="about-data-title">{escape(t("about_title"))}</div>
            <p class="about-data-text">{escape(t("about_text"))}</p>
        </div>
        """
    )

    # Filter pills: "All" (the default) shows every question in order; the other pills
    # narrow the list to one group. The options are fixed ids; format_func shows them in
    # the current language, so switching language keeps the chosen filter.
    faq_filters = ["all", "model", "data", "support"]
    st.session_state.setdefault("faq_group", "all")   # so the pill stays highlighted after a language switch
    with st.container(key="pills_faq"):
        if hasattr(st, "segmented_control"):
            chosen_filter = st.segmented_control(
                "Filter the questions by topic",
                faq_filters,
                format_func=lambda option: t(f"filter_{option}"),
                key="faq_group",
                label_visibility="collapsed",
            )
        else:  # older Streamlit versions: same behavior, plainer look
            chosen_filter = st.radio(
                "Filter the questions by topic",
                faq_filters,
                format_func=lambda option: t(f"filter_{option}"),
                horizontal=True,
                key="faq_group",
                label_visibility="collapsed",
            )
    chosen_filter = chosen_filter or "all"     # segmented_control returns None if un-clicked

    # Every question is (number, group). The texts live in TEXT as faq_q<number> and
    # faq_a<number>, kept in the order they are shown.
    faq_order = [
        (1, "model"), (2, "model"), (3, "model"), (4, "model"),
        (5, "data"), (6, "data"), (7, "data"),
        (8, "support"), (9, "support"), (10, "support"), (11, "support"),
    ]
    faq_values = dict(
        test=TEST_SET_SIZE,
        rows=number_of_rows,
        r2=MODEL_R2,
        r2p=f"{MODEL_R2:.0%}",
        mae=MODEL_TYPICAL_ERROR_SAR,
        raw=RAW_LISTINGS,
        kept=f"{kept_share:.0%}",
        removed=removed_total,
        neg=REMOVED_NEGOTIABLE,
        dup=REMOVED_DUPLICATES,
        imp=REMOVED_IMPLAUSIBLE,
        make=isolate(str(most_common_make)),
        n=most_common_make_count,
        p=most_common_make_share,
    )

    # Show the questions that match the chosen pill, each as its own independent expander
    visible_faq = [(number, group) for number, group in faq_order if chosen_filter in ("all", group)]
    if chosen_filter != "all":
        show_html(
            '<div class="faq-count">'
            + escape(
                t("faq_count", n=len(visible_faq), m=len(faq_order), group=t(f"filter_{chosen_filter}"))
            )
            + "</div>"
        )
    for number, _group in visible_faq:
        with st.expander(t(f"faq_q{number}")):
            st.markdown(t(f"faq_a{number}", **faq_values))
