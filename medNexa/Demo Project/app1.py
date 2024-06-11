from flask import  Flask
## i will create a simple flask application

app1 = Flask(__name__)
# flask application variable is app and flask(__name__) is entry point of program


if __name__ == '__main__':
    app1.run(debug=True)

# degug is not true then if we update some code then we need to stop sever
