import express from "express";
import { spawn } from "child_process";

const app = express();
app.use(express.json());

app.get("/hello/", (req, res) => {
  const { driveLink } = req.body;

  const pythonScript = spawn("python", [
    "-u",
    "pyModels/yunet_v_5.1.0.py",
    driveLink,
  ]);

  pythonScript.stdout.on("data", (data) => {
    // console.log(`Python script output: ${data}`);
    res.send(`Python script is running with Drive CSV Link: ${data}.`);
  });

  pythonScript.stderr.on("data", (data) => {
    console.error(`Error executing Python script: ${data}`);
  });

  pythonScript.on("close", (code) => {
    console.log(`Python script exited with code ${code}`);
  });
});

app.listen(5001, () => {
  console.log("Listening on Port 5001");
});
