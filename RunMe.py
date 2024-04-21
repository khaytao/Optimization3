import sys
from main import question_16, question_17


def q16():
    print("===================")
    print("Question 16:")
    print()

    w, w0, lamda = question_16()
    print("The coefficients 'w' are:")
    print(w)
    print(f" 'w0' is: {w0}")
    print("The dual coefficients 'lambda' are:")
    print(lamda)

    print()
    print("===================")

def q17():
    print("===================")
    print("Question 17 (Using kernel):")
    print()

    w, w0, lamda = question_17()
    print("The coefficients 'w' are:")
    print(w)
    print(f" 'w0' is: {w0}")
    print("The dual coefficients 'lambda' are:")
    print(lamda)

    print()
    print("===================")


def main():
    """All answers to code questions."""
    q16()
    q17()


# FLags definition
if len(sys.argv) > 1:
    flag = sys.argv[1]

    flags = {
        "-q16": q16,
        "-q17": q17,
        "-main": main
    }

    flags[flag]()
