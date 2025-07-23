from flask import Flask, request, render_template
from pymongo import MongoClient

# Create the Flask object
app = Flask(__name__)

# Endpoints are modeled as "routes". We create one for /. We
# specify access methods as a list to the methods parameter.

@app.route('/', methods = ['GET'])
def root():
    return 'SWS3009 Sample Site', 200

# Examples of how to render a template. Also note
# how we use requests.args.get to extract GET parameters
@app.route('/index', methods = ['GET'])
def index():
    """ Demo routine to show how to pass parameters through GET """

    # Extract GET parameters from request object
    name = request.args.get('name')

    if name is None:
        name = 'Bob Jones'

    return render_template('index.html', info = {"title":"Hello World", "name":name}), 200

# Example of how to handle JSON sent in via POST
@app.route('/put', methods = ['POST'])
def add():
    """ Add a new record to the database """

    try:
        new_rec = request.get_json()

        print(new_rec)
        if new_rec is not None:
            col.insert_one(new_rec)

        return 'OK', 200
    except Exception as e:
        return e, 400

    
@app.route('/get', methods = ['GET'])
def get():
    """ Get all records and return it """

    results = col.find()

    return render_template('get.html', results = results)

# Main code

def main():
    global client, db, col, app
    client = MongoClient('mongodb://localhost:27017/')
    db = client['my_db']
    col = db['MyCollection']
    app.run(port = 3237, host="0.0.0.0")

if __name__ == '__main__':
    main()

