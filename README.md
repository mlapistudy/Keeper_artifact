# Readme
This is the artifact for our TOSEM paper "Keeper: Automated Testing and Fixing of Machine Learning Software". It is publiclty available at [GitHub](https://github.com/mlapistudy/Keeper_artifact). It is also archived at [Zendo](https://zenodo.org/records/10968650).

Keeper is a systematic testing and fixing tool for ML software. It is an extended work from our [ICSE 22](https://github.com/mlapistudy/ICSE2022_158) work.


## How to install
Please follow the instructions in  `REQUIREMENTS.md`

## How to use CMD version.
It is under folder `./testing_tool`. Please check `./testing_tool/readme.md` for details.

If it shows a malloc error, please launch it in a seperate terminal outside the IDE.

## How to launch IDE plugin
Open the **`./Keeper/` folder** in VS Code. Please make sure it is not the parent/child folder of `Keeper/`, otherwise VS Code would not able to parse the project.

Then select `./src/extension.ts`. Click "run" -> "start debugging" on top menu or pressing F5. Then the plugin interface would appear in a new VS Code window. 

Then, set up environment and application prerequisites. This is the line that gets executed before any of our analysis file runs. Typically, one should include (1) export the google cloud credentials; (2) export the path to CVC4; (3) any other commands needed to activate a virtual environment (e.g. anaconda) for the particular python environment we require, etc. This can be done by going to the Settings in VS Code (details can be found on the VS Code documentations here), search for Mlapi-testing: Set Up Environment And Application Prerequisites, and modify the entry to include these. An example of including (1) and (2) is:

``` bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/google/credential.json'; export PYTHONPATH=/usr/local/share/pyshared/;
```

![Snapshot](demo/settings.png)


We provide an example input in `./plugin_example`. To use this example, please open this folder in the new VS Code window.


## How to use the plugin interface
1. Click on the plugin icon on the left side of your screen to reveal the plugin window. It may take several seconds.
![demo1](demo/demo1.jpeg)
2. Next, click on the refresh button in the upper right hand corner of the plugin window, or the "Detect Relevant Functions" button in the bottom third of the plugin window, in order to find functions that can be tested by our plugin.
![demo2](demo/demo2.jpeg)
3. Next, click on the function you want to test and click on the button "Test This Function" located to the right of the function name. You can also input information for a function not shown in the plugin window by clicking on the "Input for testable functions" button.
![demo3](demo/demo3.jpeg)
4. Next, for each of the selected function's parameters, fill out what type the parameter is and whether it is used in a Machine Learning Cloud API.
![demo4](demo/demo4.jpeg)
5. Once the types have been inputted you will see a pop-up window where you can click the "Log Messages" button. Clicking this button will allow you to see the progress of our tool while it runs. Depends on the network and number of test cases, it may take several minutes to execute.
![demo5](demo/demo5.jpeg)
6. Congrats! Right under the view for the testable functions you will see information about any bugs or inefficiencies your selected function has. You will also see the lines of code with bugs underlined for you! If you want to remove the underlines, click the "Remove underlines" button.
![demo6](demo/demo6.jpeg)

## GIF demo
![video demo](demo/demo-video.gif)


## Paper result
It is under branch `paper-result`. Please check `./readme.md` of this branch for details.
