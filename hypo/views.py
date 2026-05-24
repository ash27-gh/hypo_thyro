from django.shortcuts import render
import pandas as pd
import joblib
def index(request):
    return render(request,'index.html')


def predict(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        sex = int(request.POST.get('sex'))
        tsh = float(request.POST.get('tsh'))
        t3 = float(request.POST.get('t3'))
        t4 = float(request.POST.get('t4'))
        t4u = float(request.POST.get('t4u'))
        fti = float(request.POST.get('fti'))
        tsh_t3 = float(request.POST.get('tsh_t3_ratio'))
        tsh_t4 = float(request.POST.get('tsh_t4_ratio'))
        t4_t3 = float(request.POST.get('t4_t3_ratio'))
        age_tsh = float(request.POST.get('age_tsh_interaction'))
        features = ['Age', 'sex', 'TSH', 'T3', 'T4', 'T4U', 'FTI', 'TSH_T3_ratio', 'TSH_T4_ratio',
                    'T4_T3_ratio', 'Age_TSH_Interaction']
        user_input = [age,sex,tsh,t3,t4,t4u,fti,tsh_t3,tsh_t4,t4_t3,age_tsh]

        model = joblib.load('Hypothyroid_model.joblib')
        scaler=joblib.load("scaler.joblib")
        input_array = pd.DataFrame([user_input], columns=features)
        input_array_scaled = scaler.transform(input_array)
        proba = model.predict_proba(input_array_scaled)[0][1]

        threshold = 0.3
        prediction = 1 if proba > threshold else 0
        print("Predicted probability of Hypothyroid:", round(proba, 3))
        result="Predicted Result:" + "Hypothyroid" if prediction == 1 else "Not Hypothyroid"
        return render(request, 'result.html',{"result":result})


    return render(request,'predict.html')