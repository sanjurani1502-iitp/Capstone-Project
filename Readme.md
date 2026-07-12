# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis 

# Project overview

* Is project me maine ek "raw Car Price Dataset" liya, kyuki car market ek aisi cheez hai jise har koi aasani se samajh sakta hai, Brand, fuel type (Petrol/Diesel), engine specifications, aur kilometers driven—ye saari aisi cheezon hain jo hum normal life me dekhte hain, Ek data professional ke taur par practice karne ke liye hame aisa dataset chahiye hota hai jisme bohot saari galtiyan hon taaki hum unhe theek karna seekh sakein. Is raw car dataset me hume sab kuch mila:

* Is ek dataset ne hamari pooro data cleaning ki skills ko test kar liya.

* uske andar ke gap ko bhara, aur data ki visheshtayen ko samjha

* Iska maqsad machine learning modeling ke liye ek high-quality aur error-free dataset taiyar karna h.

## Short me kahein toh: ye dataset isliye chuna kyunki ye ek perfect "Practice Ground" hai jo industry-level ke saare data cleaning, machine learning, aur real-world problems ko ek sath cover karta hai!

# data ki jaanch aur safai
* null value: kilometers_driven aur seats me missing value payi thi, inhe meadian se fill kiya.

# Justification
* Median ka use isliye kiya kyunki hamara data skewed hai (outlier hai),
Mean (average) outliers se affect ho jata hai, jabki Median data ke center ko sahi represent karta hai. 

# duplicate
* Dataset me 15 duplicate row thi, drop_duplicates() ka use karke remove kiya Isse data ki quality improve hui.

* Engine_CC - jo ki string format me thi, use number me convert kiya, taki culculation kar sake, 
* "Car_Brand aur Fuel_Type" ko Category dtype me badala taki memory kam use ho.

# Skewness
* Kilometers_Driven aur Price mein high skewness dekhi gayi.
* jo mean ko right side khich rahi hain. Is wajah se imputation ke liye Median hi best option tha.

# Outlier Detection
* Inter-Quartile Range ka use karke outlier find kiya, 
* outliers ko drop nahi kiya, kyunki ye car price ka real data hai
* modeling ke wakt inhe handel karne ke liye scaling ka use karenge.

# Visualizations
* line_plot - Price mein variation dikhata hai.
* Bar chart - Car_Brand ke hisaab se average price compare kiya.
* Histogram - Kilometers_Driven ka distribution show karta hai .
* scatter plot - Engine_CC aur Price ke beech relationship dikhata hai
* Box Plot - Fuel_Type ke hisaab se Price distribution mein difference dikhata hai.

# correlation
* Pearson vs Spearman -  maine dono correlation check kiye. Pearson linear relationship batata hai, jabki Spearman non-linear monotonic trends ko pakadta hai.
* top 3 Differences: Jin pairs ka difference sabse zyada hai, wahan linear method (Pearson) kamzor padta hai. Isliye feature selection ke liye hum Spearman correlation ko prefer karenge, kyunki ye outliers se affect nahi hota aur non-linear relationships ko acche se samajhta hai.

# Grouped aggregation.
* Highest Mean: Hyundai brand ki cars ka average price sabse zyada hai.
* Highest Std Dev: Tata mein price mein variation sabse zyada hai.

* Ratio: Max/Min ratio humne calculate kiya, jo ye batata hai ki Car_Brand feature ka price prediction mein kaafi bada role hai.

* Conclusion: High standard deviation ka matlab hai ki sirf brand dekh kar price predict karna risky hai, hame Engine_CC aur Year ko bhi include karna hoga.


# Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate


* Is project me ek car dataset (cleaned_car_data.csv)ka use kiya h.
* yah notebook bina kisi error ke top to bottom  run hoti h.

* Is project me "cleaned_car_dataset.csv" data ka use karke do alag-alag type ke predictive models banaye hai, ek Regression Model aur ek Binary Classification Model, code ko poori tarah se data leakage se bachaya gaya hai aur final results ko standard metrics ke sath evaluate kiya hai.

# target variable definition & categorical encoding
* Regression target (y_reg) - Isme maine continuous column Price ko target mana hai.
* Classification Target (y_clf) - Isko maine Price column ke median par binarize karke banaya hai.

* class 0 - agar car ki price median price se kam ya uske barabar hai.
* class 1 - agar car ki price median price se zyada hai.

* dataset mein teen categorical features hai, Car_Brand, Fuel_Type, aur Transmission

# One-Hot Encoding Kyun Use Kiya
* in teen colomn me koi natural order nahi hai, Petrol aur Diesel mein koi chhota-bada nahi hota, 
* Agar yahan Label Encoding karta, toh model unhe galat sequence (1, 2, 3) mein samajh leta aur predictions kharab ho jatein. Isliye pd.get_dummies() use karke dummy variables banaye.

* code mein drop_first=True specify kiya hai taaki har category ka pehla dummy column drop ho jaye.

# Data Split aur Leak-Free Scaling
* Maine train_test_split use karke data ko 80% Training set aur 20% Test set mein baanta hai, aur random_state=42 lagaya * hai taaki har baar same split mile.

* Maine StandardScaler ko sirf training data (X_train) par .fit() kiya hai, aur test data (X_test) ko sirf .transform() kiya hai. Agar poore dataset par ek sath scaler fit kar deti, toh test set ka mean aur variance training ke time par hi leak ho jata. Isse model ko pehle se test data ka idea lag jata, jise Data Leakage kehte hain.

# Regression Model — Linear vs. Ridge Regression
Model train karne ke baad jab coefficients ki absolute values ko sort kiya, toh top 3 features yeh nikal kar aaye:
1. Car_Brand_Maruti (Coefficient: 35195.92)
2. Car_Brand_Hyundai (Coefficient: 32343.02)
3. Car_Brand_Tata (Coefficient: 25532.53)

* Large Positive Coefficient ka matlab: Agar baki saare features same rahein, toh is scaled feature ki value 1 unit badhne se car ki predicted price units badh jayegi.

* Large Negative Coefficient ka matlab: Is scaled feature ki value 1 unit badhne se car ki predicted price utne units kam ho jayegi.

# model evaluation camparison
* OLS Linear Regression : MSE-(MSE: 125348639738.88), R-squared (R2) - R2: -0.0147
* Ridge Regression (a = 1.0): MSE -(MSE: 125349781454.17), R-squared (R2) - R2: -0.0147

# Ridge aur OLS mein difference kyun aaya
* OLS model sirf training loss kam karne ke chakkar mein kabhi-kabhi coefficients ko bohot bada bana deta hai. Ridge Regression ek L_2 penalty lagata hai

# Classification Model — Logistic Regression , Class Imbalance Check karna
* Model chalane se pehle maine training set ka distribution check kiya
Class 0:[0.501923]
Class 1: [0.498077]

* kyu ki target ko median ke aas pass binarize kiya tha, isliye dono classes ka distribution pehle se hi balanced hai, Is wajah se mujhe SMOTE ya class_weight='balanced' lagane ki koi zaroorat nahi padi, data naturally perfect tha.

# evaluation metrics
* maine basline model (C=1.0) par Confusion Matrix aur Classification Report generate kiya, jisse Precision, Recall aur F1-score mile. Sath hi, ROC Curve plot karke AUC Value: [51%] nikali. jo yeh dikhati hai ki model high aur low price wale cars ko kitne acche se separate kar pa raha hai.

# Decision-Threshold Sensitivity Analysis
* Maine default threshold (0.5) ko badal kar 0.30 se 0.70 tak test kiya taaki dekh sakun ki Precision aur Recall par kya asar padta hai.


Classification Report ---
              precision    recall  f1-score   support

           0       0.52      0.36      0.43        66
           1       0.50      0.66      0.57        64

    accuracy                           0.51       130
   macro avg       0.51      0.51      0.50       130
weighted avg       0.51      0.51      0.50       130

# Decision-threshold sensitivity
* F1-Maximizing Threshold: model threshold 0.30 par sabse zyada F1-Score (0.66) de raha hai, jo Precision aur Recall ka ek achha balance hai.

* Lower Threshold (0.40): Ye Recall ko badha deta hai.
* Higher Threshold (0.60): Ye Precision ko badha deta hai.

# Regularization experiment on Logistic Regression:
* Logistic Regression mein C parameter inverse regularization strength c=1 hota h,
c=1.0 Regularization kam hoti hai, jisse model patterns ko ache se seekh sakta hai.
C=0.01 ye model ke weights par bohot kadi penalty lagata hai aur unhe chhota kar deta hai taaki overfitting na ho.

# Bootstrap confidence interval for AUC difference.
* maine dono models (C=1.0 aur C=0.01) ke auc ki reliability check karne ke liye np.random.choice se 500 bootstrap samples (with replacement) liye:

* Mean AUC Difference: (c=1.0 - c=0.01): -0.0058
* 95% Confidence Interval: [-0.0058, -0.0058]

* Hamare result mein range [-0.0058, -0.0058] freeze ho gayi hai, jisme 0 nahi aata hai. Lekin dono number bilkul ek jaise hona sample size ya code calculation ki limitation ko darshata hai, isliye ise poori tarah reliable nahi mana ja sakta.



# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

* Is part me model ko aur advanced banaya hai, ensembles Random Forest & Gradient Boosting ka use kiya hai, parameters ko tune kiya hai, aur ek robust reproducible Pipeline taiyar ki hai.

# decision tree Baseline vs Controlled Tree
* Unconstrained Decision Tree: Is model me tree ki growth par koi constraint nahi lagaya.

* overfiting ki pahchan: Agar output training accuracy ko bohut high(1.00)dikhata hai,aur test accuracy kafi kam aati hai, toh iska matlab model Overfitting kar raha hai.

* High Variance Kyun?: Decision trees ko high-variance model isliye kaha jata hai kyunki ye har split par bina soche-samjhe greedy tarike se training data ko perfectly fit karne ki koshish karte hain

# Controlled Decision Tree
* max_depth=5 gadiyon ke splits ko limit karta hai.
* min_samples_split=20 tabhi split allow karta hai jab node mein minimum 20 samples hon, jisse chote data subsets ka noise avoid hota hai.
Result: Is controlled tree ka Train aur Test accuracy gap, unconstrained tree ke muqable kaafi kam aur stable ho jata hai.

# Gini vs Entropy Comparison
isme max_depth=5 ke sath do alag criterion test kiye.Dono ke Mathematical Formulas ye hain: 

* Gini Impurity Formula: $$\text{Gini} = 1 - (p_1^2 + p_2^2 + \dots + p_n^2)$$

* Entropy Formula: $$\text{Entropy} = - (p_1 \log_2 p_1 + p_2 \log_2 p_2 + \dots + p_n \log_2 p_n)$$

* Gini = 0 ka matlab: Jab kisi node ka Gini impurity score 0 ho jata hai, iska matlab hai ki woh node Pure hai—yani us node ke andar maujood saari cars sirf aur sirf ek hi class (ya toh sab low-price ya sab high-price) se belong karti hain.

# Random Forest
Isme n_estimators=100 aur max_depth=10 ke sath Random Forest train kiya.
* Feature Importance: Random Forest yah dekhta hai ki koi feature sare trees me Gini impurity ko avrage kitna kam kar raha h, yah Linear Regression ke coefficients se alag hai, kyuki yah non-linear impact ko bhi kam kar leta h.

* Bagging Concept: Iska matlab hai ki har tree training data ka ek random sample (with replacement) dekhta hai aur har split par sirf $\sqrt{\text{total features}}$ ka ek random subset hi consider kiya jata hai. In sabhi trees ka average nikalne se variance kam ho jata hai aur model robust banta hai.

# Feature Ablation Study
* Maine Random Forest ke outputs se 5 sabse kam importance wale features (lowest_5_features) ko pehchana aur unhe data se poori tarah drop karke ek reduced model train kiya:

* Maine Random Forest ke hisab se 5 sabse kam importance wale features ko drop karke jab reduced model check kiya, toh Test ROC-AUC par koi bada negative asar nahi pada. Iska matlab ye features uninformative the aur sirf noise add kar rahe the. Production ke point of view se, in features ko hatane se hamari pipeline simple ho jayegi, inference cost (prediction speed) fast hogi aur maintenance ka burden bhi kam ho jayega

# Cross-Validated Comparison
* Isme StratifiedKFold(n_splits=5) ka use karke sabhi main models (Logistic Regression, Controlled DT, Random Forest, Gradient Boosting) ka Mean AUC aur Std AUC nikala.

* Cross Validated Kyun Reliable Hai? Cross-validation ek single train-test split se kahin zyada reliable estimate deta hai kyunki ye poore data ko 5 alag-alag folds mein baantkar har hisse ko test set banne ka mauka deta hai. Isse hume model ki asli stability aur generalization performance ka pata chalta hai.

# Hyperparameter Tuning with GridSearchCV
* Raw unscaled X_train ka use karke ek robust end-to-end pipeline banayi gayi (SimpleImputer + StandardScaler + RandomForestClassifier). Pipeline ke andar parameters set karne ke liye double underscore (__) syntax ka use kiya hai.

* Isme total **90 model configurations** evaluate huye hain. Exhaustive Grid Search har ek combination ko check karta hai jabki Randomized Search kam time me random setups test karke jaldi results deta hai.

# Manual learning curve
* Tuned pipeline ko data ke progressively larger subsets (20%, 40%, 60%, 80%, 100%) par fit kiya gaya, aur training fraction ke hisab se AUC evaluate kiya hai.

**Conclusion:** Jaise-jaise training data badhta hai, training AUC thoda decrease hota hai aur test AUC increase hota hai. Agar test AUC plateau nahi hua hai, toh iska matlab model **Data-limited** hai (yani aur data badhane se performance badhegi).

# Serialize the best model:
* maine final tuned pipeline ko best_model.pkl naam se save kiya hai. Reload karne aur inference execute karne ke liye ye 5-line block perfectly run hota hai:



## Part 4 — LLM-Powered Feature: Structured Extraction, Tabular Batch Scoring, or Model Prediction Explanation

* LLM powered Feature ke liye  Model Prediction Explanation Pipeline choose kiya h.
Is section me ek end-to-end pipeline build kiya hai jo Machine Learning model ke predictions ko human-readable, structured JSON format me explain karta hai. Isme advanced prompt engineering, strict JSON schema validation, aur PII guardrails ka use kiya gaya hai.

* Humari pipeline OpenRouter API infrastructure ka use karti hai jo kisi bhi OpenAI-compatible endpoint ke saath easily scale ho sakti hai.

* Zero Hardcoding Principle: API Security maintain karne ke liye API key ko kabhi bhi code me hardcode nahi kiya gaya hai. System secure management ke liye environment variables ka use karta hai:

* api_key = os.environ.get('LLM_API_KEY') *

* Target Endpoint URL: Hum standard production route ka use kar rahe hain:
# https://openrouter.ai/api/v1/chat/completions

* ek robust connection wrapper call_llm() build kiya hai jo custom timeouts aur JSON payload encapsulation handle karta hai:

*  Payload Structure Template
payload = {
    "model": "meta-llama/llama-3-8b-instruct:free",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "temperature": temperature,
    "max_tokens": max_tokens
}

* Handshake Verification Test: Pipeline start hone par system ek automatic baseline check run karta hai ("Reply with only the word: hello"). Is request par target server se HTTP Status Code 200 milte hi pipeline dynamic data processing ke liye green signal deti hai

* Humne LLM se strictly structured output lene ke liye Zero-Shot System Prompt aur ek dynamic User Prompt Template ka use kiya hai.

* Verbatim Prompts
* System Prompt
* You are an expert Machine Learning Model Interpreter. Analyze the provided feature values, predictions, and probabilities to generate a structured explanation. You must respond with ONLY a valid JSON object. Do not include markdown blocks, text wrappers, or trailing commentary. The JSON response must match this schema strictly: {"prediction_label": "string", "confidence_level": "low|medium|high", "top_reason": "string", "second_reason": "string", "next_step": "string"}

* User Prompt Template
Model Inputs provided for evaluation:
- Brand, Engine Specs, Features: {features}
- Predicted Category (1 for > Median Price, 0 for <= Median): {pred_class}
- Confidence Probability Score: {pred_prob:.4f}
Generate the validated explanation JSON payload matching the target template criteria.

# Choosing Temperature = 0
maine core production pipeline ke liye temperature=0.0 select kiya hai kyunki hume deterministic aur predictable outputs chahiye. Low temperature par model hamesha highest-probability wale next token ko select karta hai, jisse random creativity kam hoti hai aur strict JSON schema structure break nahi hota. Production pipelines me data integrity banaye rakhne ke liye yeh zaroori hai.

# Temperature A/B Comparison

Humne ek test record (BMW, Engine: 2500CC) ko dono temperature=0 aur temperature=0.7 par run karke assess kiya hai:

| Input (Feature Specs) | Output at `temp=0` | Output at `temp=0.7` | Key Difference |
| :--- | :--- | :--- | :--- |
| BMW, 2500CC, 2021, Petrol, Auto | 
`{"prediction_label": "High Price Variant", "confidence_level": "high", "top_reason": "Engine capacity is highly configured at 2500CC.", "second_reason": "Car Brand falls under luxury tier classification.", "next_step": "Recommend personalized finance packages to client."}` | `{"prediction_label": "Premium Luxury Vehicle", "confidence_level": "high", "top_reason": "Equipped with a powerful 2500CC petrol engine.", "second_reason": "BMW brand premium positioning.", "next_step": "Offer exclusive test drive and VIP financing options."}` | `temp=0` par model strictly greedy decoding use karta hai, jabki `temp=0.7` par words/synonyms me variability aur creativity badh jati hai. |


* Why the difference? temperature=0 par model strictly greedy decoding use karta hai, yani sabse zyada probability wale words choose karta hai. Jab hum temperature badhakar 0.7 karte hain, toh probability distribution flatten ho jati hai, jisse model thode kam probable par zyada diverse aur creative words/synonyms ko sample kar pata hai. Isse variability badh jati hai.

* Guardrails: PII Protection Verification
Data privacy ensure karne ke liye, LLM API call se theek pehle ek Regex-based Guardrail check lagaya gaya hai jo kisi bhi Personally Identifiable Information (PII) jaise Email ya Phone Number ko block karta hai.

# End-to-End Pipeline & Validation Results
Humne final pipeline ko 3 diverse inputs par test kiya. Agar LLM ka output JSON valid nahi hota ya schema criteria (jsonschema.ValidationError) fail karta hai, toh pipeline automatically ek predefined FALLBACK_JSON return karti hai jisme mandatory fields null ho jate hain.


* Humari pipeline successfully har valid response ko parse karti hai. Humne jsonschema ka validation engine use kiya hai jo yeh check karta hai ki:

* Output ek valid JSON object ho (json.JSONDecodeError handling).

* Saari 5 required scalar fields (prediction_label, confidence_level, top_reason, second_reason, next_step) usme maujood hon.

confidence_level strictly ["low", "medium", "high"] ke enum set se hi match kare.


##  How to Run the Project (Quick Start)

Is pooray project ko reproduce ya run karne ke liye niche diye gaye steps follow karein:

1. **Dependencies Install Karein:**
   ```bash
   pip install pandas numpy scikit-learn jsonschema openrouter

# Environment Variable Set Karein (Part 4 ke liye):
* Linux/Mac: export LLM_API_KEY="your_api_key_here"   
* Windows (CMD): set LLM_API_KEY="your_api_key_here"

# Pipeline Execute Karein:
* Saari notebooks ya scripts cleaned_car_data.csv ka use karke top to bottom bina kisi error ke run ho jayengi. Final tuned model best_model.pkl naam se serialize ho jayega.


# Conclusion
* Is project se humne ek end-to-end robust ML workflow banaya hai. Jahan raw data cleaning se lekar advanced non-linear ensemble models (Random Forest) tak ka safar tay kiya, aur aakhiri me ek modern LLM layer jodi jo model ke decisions ko transparent aur human-readable banati hai. Ye pipeline production-ready hai aur real-world car valuation systems me use ki ja sakti hai.

