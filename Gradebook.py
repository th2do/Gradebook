""" Calculate students' grades by combining data from many sources.
 
 Using pandas, this script combines data from the:

 *Roster
 *Homework & Exam Grades
 *Quiz grades

 to calculate the final grades for the class
"""

from pathlib import Path 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy

HERE = Path(__file__).parent 
DATA_FOLDER = HERE 

#read data in
roster = pd.read_csv(
    DATA_FOLDER / "roster.csv", 
    converters = {"NetID":str.lower, "Email Address": str.lower},
    usecols = ["Section", "Email Address", "NetID"],
    index_col  = "NetID"
)

hw_exam_grades = pd.read_csv(
    DATA_FOLDER / "hw_exam_grades.csv",
    converters = {"SID": str.lower},
    usecols = lambda x: "Submission" not in x, 
    index_col = "SID"
)

quiz_grades = pd.DataFrame()
for file_path in DATA_FOLDER.glob("quiz_*_grades.csv"):
    quiz_name = " ".join(file_path.stem.title().split("_")[:2])
    quiz = pd.read_csv(
        file_path,
        converters = {"Email": str.lower},
        index_col = ["Email"],
        usecols = ["Email", "Grade"],
    ).rename(columns = {"Grade": quiz_name})
    quiz_grades = pd.concat([quiz_grades, quiz], axis = 1)

#combining all the datas into one dataframe
final_data = pd.merge(roster, hw_exam_grades, left_index = True, right_index = True)
final_data = pd.merge(final_data, quiz_grades, left_on = "Email Address", right_index = True)
final_data = final_data.fillna(0)

#Calculating the Exam Total Score
n_exams = 3
for n in range(1, n_exams + 1):
    final_data[f"Exam {n} Score"] = (
        final_data[f"Exam {n}"] / final_data[f"Exam {n} - Max Points"]
    )

#Calculating the Homework Scores
hw_scores = final_data.filter(regex = "^Homework \d\d?$", axis = 1)
hw_max_scores = final_data.filter(regex = "^Homework \d\d? -", axis = 1)

sum_of_hw_scores = hw_scores.sum(axis = 1)
sum_of_max_hw_scores= hw_max_scores.sum(axis = 1)
final_data["Total Homework"] = sum_of_hw_scores / sum_of_max_hw_scores

hw_max_renamed = hw_max_scores.set_axis(hw_scores.columns, axis = 1)
average_hw_scores = (hw_scores / hw_max_renamed).sum(axis = 1)
final_data["Average Homework"] = average_hw_scores / hw_scores.shape[1]

final_data["Homework Score"] = final_data[["Total Homework", "Average Homework"]].max(axis = 1)

#Calculating the Quiz Score
quiz_scores = final_data.filter(regex = r"^Quiz \d$", axis = 1)
quiz_max_points = pd.Series(
    {"Quiz 1": 11, "Quiz 2": 15, "Quiz 3": 17, "Quiz 4": 14, "Quiz 5": 12}
)
sum_of_quiz_scores = quiz_scores.sum(axis = 1)
sum_of_max_quiz_scores = quiz_max_points.sum()
final_data["Total Quiz"] = sum_of_quiz_scores / sum_of_max_quiz_scores

average_quiz_scores = (quiz_scores / quiz_max_points).sum(axis = 1)
final_data["Average Quiz"] = average_quiz_scores / quiz_scores.shape[1]

final_data["Quiz Score"] = final_data[["Total Quiz", "Average Quiz"]].max(axis = 1)

#Calculating the final grade
weightings = pd.Series(
    {
        "Exam 1 Score": 0.05,
        "Exam 2 Score": 0.1,
        "Exam 3 Score": 0.15,
        "Quiz Score": 0.30, 
        "Homework Score": 0.4
    }
)
final_data["Final Score"] = (final_data[weightings.index] * weightings).sum(axis = 1)
final_data["Ceiling Score"] = np.ceil(final_data["Final Score"] * 100)

grades = {
    90: "A+",
    85: "A",
    80: "A-",
    77: "B+",
    73: "B", 
    70: "B-",
    67: "C+",
    63: "C",
    60: "C-",
    0: "F"
}

def grade_mapping(value):
    for key, letter in grades.items():
        if value >= key:
            return letter 

letter_grades = final_data["Ceiling Score"].map(grade_mapping)
final_data["Final Grade"] = pd.Categorical(
    letter_grades, categories = grades.values(), ordered = True
)

#Plotting Summary Statistics

final_data["Final Score"].plot.hist(bins = 20, label = "Histogram")
final_data["Final Score"].plot.density(
    linewidth = 4, label = "Kernel Density Estimate"
)

final_mean = final_data["Final Score"].mean()
final_std = final_data["Final Score"].std()
x = np.linspace(final_mean - 5 * final_std, final_mean + 5 * final_std, 200)
normal_dist = scipy.stats.norm.pdf(x, loc = final_mean, scale = final_std)
plt.plot(x, normal_dist, label = "Normal Distribution", linewidth = 4)
plt.legend()
plt.show()