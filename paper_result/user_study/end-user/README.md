# README

This folder contains material and aggregated results of our user study with end-users.

## Materials

The preview of the survey is available at `./survey_preview.pdf`. Consent form is available at `./consent_form.pdf`

## Results

Due to IRB restriction, we only include the aggregated result of 100 participants at `./survey_result.xlsx`. It is obtained by applying `./parse_survey.py` to survey raw data. 

In `./user_study/survey_result.xlsx`, there are two tabs:

1. Result tab: Contains answer distribution of each survey question. The cells for computation are colored with light blue: Row 28-32 shows the overall preference of each application example.
2. Explanation tab: Contains explanation of each question ID:


| Question ID | Meaning                                                      |
| ----------- | ------------------------------------------------------------ |
| Q*-0        | Individual preference of each test case                      |
| Q*-1        | Which one has a higher recall? (Q3-1 asks about emotion matching of input  and output) |
| Q*-2        | How important is recall?                                     |
| Q*-3        | Which one has a higher precision?                            |
| Q*-4        | How important is precision?                                  |
| Q*-5        | Overall preference of entire software                        |

Q3 (Smart diary) recognizes emotion of the uploaded diary entry, which is a more complex question than answering T/F. So it do not have precision/recall. Instead, it asks about matching between the emotion in the response and the emotion in the uploaded diary entry.

