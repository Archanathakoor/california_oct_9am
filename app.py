
from flask import Flask, render_template, request
import joblib
import pandas as pd
import math

app = Flask(__name__)

# Load the saved model dictionary
Obj = joblib.load("california.joblib")

# IMPORTANT: Use the exact capitalization of your keys
model = Obj["Model"]
columns = list(Obj["Columns"])

print("Model loaded successfully!")
print("Feature columns:", columns)


@app.route("/", methods=["GET", "POST"])
def Main():
    prediction = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            input_values = []

            for column in columns:
                raw_value = request.form.get(column, "").strip()

                if not raw_value:
                    raise ValueError(f"Please enter {column}.")

                value = float(raw_value)

                if not math.isfinite(value):
                    raise ValueError(f"Invalid value for {column}.")

                input_values.append(value)
                values[column] = raw_value

            # Keep features in the original training order
            input_data = pd.DataFrame(
                [input_values],
                columns=columns
            )

            result = model.predict(input_data)[0]
            prediction = float(result)

            if not math.isfinite(prediction):
                raise ValueError("The model returned an invalid result.")

        except (ValueError, TypeError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Please check your model and inputs."

    return render_template(
        "index.html",
        columns=columns,
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)

