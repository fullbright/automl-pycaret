from pycaret.classification import *
import pandas as pd

model = load_model("tuned_model")

def predict_category(data_unseen):

    prediction = predict_model(model, data=data_unseen)
    print(prediction.dtypes)
    print(prediction["prediction_label"])
    print(prediction["prediction_score"])

    return prediction

data_columns=["Date","name","merchant","Montant","Categorie","Categorie2","Libellé","entree_sortie_argent","month","amount","year","fixeOuVariable","FournisseurClient","TransactionId","BudgetMontant","AnomalyAndComments","HasOverflowBudget"]
data_1 = ["11/11/2024","Achat Ali express piltover","Compte De Dépôts - Sergio Afanou",72,"","",0,"sortie",11,72,2024,"variable","","ID-03254",0,"",0]
data_2 = ['04/04/2024','Vir.permanent Afanou','Livret A',11.0,'Entrées d\'argent','Virements internes',0,'Entree',4,11,2024,'Variable','Afanou','ID-00001',200,'',0]
data_3 = ["11/11/2024","Achat Amazon","Compte De Dépôts - Sergio Afanou",72,"","",0,"sortie",11,72,2024,"variable","","ID-03254",0,"",0]
data_4 = ["11/11/2024","Achat Carrefour","Compte De Dépôts - Sergio Afanou",72,"","",0,"sortie",11,72,2024,"variable","","ID-03254",0,"",0]
data_5 = ["22/04/2024","CB Paypal *alipay E","Compte De Dépôts - Sergio Afanou",-19.0,"Dépenses pro","Services en ligne",0,"Sortie",4,-19,2024,"Variable","","ID-00021",0,"",0]


datatopredict = pd.DataFrame([data_1, data_2, data_3, data_4, data_5], columns=data_columns)
predict_category(datatopredict)