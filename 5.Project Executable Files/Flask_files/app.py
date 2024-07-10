from flask import Flask, render_template, request
import pickle

label_encoders={
    'Gender':None,'Mode_of_Shipment':None,'Product_importance':None,'Warehouse_block':None
}
for lb in label_encoders.keys():
    with open('Label_Encoder_'+lb,'rb') as f:
        label_encoders[lb]=pickle.load(f)

with open('Min_Max_Scaler','rb') as f:
    scaler=pickle.load(f)
with open('Ml_Model','rb') as f:
    ml_model=pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Define options for dropdown menus
        warehouse_options = ["A", "B", "C", "D", "E", "F"]
        shipment_options = ["Flight", "Ship", "Road"]
        product_importance_options = ["low", "medium", "high"]
        gender_options = ["M", "F"]

        return render_template(
            "index.html",
            warehouse_options=warehouse_options,
            shipment_options=shipment_options,
            product_importance_options=product_importance_options,
            gender_options=gender_options,
        )
    else:
        
        # Process form submission data
        warehouse_block = request.form.get("Warehouse_block")
        mode_of_shipment = request.form.get("Mode_of_Shipment")
        customer_care_calls = int(request.form.get("Customer_care_calls") or 0)  # Handle potential missing value
        customer_rating = float(request.form.get("Customer_rating") or 0.0)   # Handle potential missing value (assuming rating is a float)
        cost_of_product = float(request.form.get("Cost_of_the_Product") or 0.0)  # Handle potential missing value (assuming cost is a float)
        prior_purchases = int(request.form.get("Prior_purchases") or 0)       # Handle potential missing value
        product_importance=request.form.get("Product_importance")
        gender=request.form.get("Gender")
        discount_offered = float(request.form.get("Discount_offered") or 0.0)  # Handle potential missing value (assuming discount is a float)
        weight_in_gms = int(request.form.get("Weight_in_gms") or 0)             # Handle potential missing value (assuming weight is an integer)
        input_data={'Warehouse_block':warehouse_block,
                    'Mode_of_Shipment':mode_of_shipment,
                    'Customer_care_calls':customer_care_calls,
                    'Customer_rating':customer_rating,
                    'Cost_of_the_Product':cost_of_product,
                    'Prior_purchases':prior_purchases,
                    'Product_importance':product_importance,
                    'Gender':gender,
                    'Discount_offered':discount_offered,
                    'Weight_in_gms':weight_in_gms}
        #encoding data
        for lb in label_encoders.keys():
            input_data[lb]=label_encoders[lb].transform([input_data[lb]])[0]
            print(input_data[lb])
        input_data_lst=[]
        for lb in ['Warehouse_block','Mode_of_Shipment','Customer_care_calls','Customer_rating','Cost_of_the_Product','Prior_purchases','Product_importance','Gender','Discount_offered','Weight_in_gms']:
            input_data_lst+=[input_data[lb]]
        
        input_data_lst=scaler.transform([input_data_lst])

        # Calculate product of numeric inputs
        #product = customer_care_calls * customer_rating * cost_of_product * prior_purchases * weight_in_gms * (100 - discount_offered)
        result=ml_model.predict(input_data_lst)[0]
        if result:
            result='Yes'
        else:
            result='No'
        print([warehouse_block,mode_of_shipment,customer_care_calls,customer_rating,cost_of_product,prior_purchases,gender,discount_offered,weight_in_gms ])
        '''
        # Display submitted data and product calculation
        return f"""
        <h1>Submitted Data</h1>
        <ul>
            <li>Warehouse Block: {warehouse_block}</li>
            <li>Mode of Shipment: {mode_of_shipment}</li>
            <li>Customer Care Calls: {customer_care_calls}</li>
            <li>Customer Rating: {customer_rating}</li>
            <li>Cost of the Product: {cost_of_product}</li>
            <li>Prior Purchases: {prior_purchases}</li>
            <li>Product Importance: {request.form.get('Product_importance')}</li> <li>Gender: {request.form.get('Gender')}</li> <li>Discount Offered: {discount_offered}</li>
            <li>Weight in Grams: {weight_in_gms}</li>
        </ul>
        <p><b>Product of Numeric Inputs: {product:.2f}</b></p> """
        '''
        return render_template("output.html",
                       warehouse_block=warehouse_block,
                       mode_of_shipment=mode_of_shipment,
                       customer_care_calls=customer_care_calls,
                       customer_rating=customer_rating,
                       cost_of_product=cost_of_product,
                       prior_purchases=prior_purchases,
                       product_importance=request.form.get('Product_importance'),
                       gender=request.form.get('Gender'),
                       discount_offered=discount_offered,
                       weight_in_gms=weight_in_gms,
                       result=result
        )

if __name__ == "__main__":
    app.run(debug=True)