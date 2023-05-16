import express from "express";
import { spawn } from "child_process";

const app = express();
app.use(express.json());

app.get("/hello/", (req, res) => {
  const { url } = req.body;

  const pythonScript = spawn("python", [
    "-u",
    "pyModels/yunet_v_5.1.1.py",
    url,
  ]);

  pythonScript.stdout.on("data", (data) => {
    console.log(`Python script output: ${data}`);
  });

  pythonScript.stderr.on("data", (data) => {
    console.error(`Error executing Python script: ${data}`);
  });

  pythonScript.on("close", (code) => {
    console.log(`Python script exited with code ${code}`);
    res.send(`Python script is running with Drive CSV Link: ${url}.`);
  });
});

app.listen(5001, () => {
  console.log("Listening on Port 5001");
});
