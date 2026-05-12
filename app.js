// Saare Module import kiya-
const express = require('express'); //Web server banane ke liye
const path = require('path'); //for file directories
const { spawn } = require('child_process'); //Python script chaalne ke liye 
const expressLayouts = require('express-ejs-layouts'); //ejs layout ke liye 
const multer = require('multer');  //temporary file storage 

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.urlencoded({ extended: true }));  //forms & json data ko parse krne ke liye (complex html code handels )
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// EJS setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(expressLayouts);
app.set('layout', 'layout');

// Multer memory storage
const storage = multer.memoryStorage();
const upload = multer({ storage }).fields([
  { name: 'media', maxCount: 1 }
]);

// Routes
//Home Page
app.get('/', (req, res) => res.render('index'));
//Forecast input form-
app.get('/forecast', (req, res) => res.render('forecast'));
//Prediction logic
app.post('/predict', upload, (req, res) => {
// const { caption, hashtags, postTime, region, engagementGoal } = req.body;
const { caption, hashtags, postTime, region, engagementGoal, followers } = req.body;
  // Parse datetime
  const datetime = new Date(postTime);
  const dayOfWeek = datetime.toLocaleString('en-US', { weekday: 'long' });
  const timeOfDay = datetime.getHours();

  // Fake/fixed values for followers, likes, comments, etc. (replace with real logic if needed)
  // const { caption, hashtags, postTime, region, followers, engagementGoal } = req.body;

  const inputData = {
    followers: parseInt(followers) || 10000,
    caption: caption || "",
    hashtags: hashtags || "",
    postTime: postTime || new Date().toISOString(),
    region: region || "US",
  
    // ✅ ADD THESE
    post_type: "image",
    day_of_week: dayOfWeek,
    time_of_day: timeOfDay,
  
    ...(engagementGoal ? { engagementGoal: parseInt(engagementGoal) } : {})
  };

  const py = spawn('python', ['reach_predictor.py', JSON.stringify(inputData)]);
  let result = '';

//Python script ne kya output diya terminal me.
  py.stdout.on('data', (data) => {
    result += data.toString();  //Kyunki data chunk-wise aata hai, isliye += use karke saara data jod rahe ho
  });
// Agar Python script me koi error aayi 
//(jaise exception), to wo yaha catch hoga.
  py.stderr.on('data', (data) => {
    console.error(`Python error: ${data}`);
  });

  py.on('close', (code) => {
    try {
      const prediction = JSON.parse(result);
      res.render('result', { result: prediction });
    } catch (err) {
      console.error(err);
      res.render('result', { result: { reach: 'Error parsing prediction' } });
    }
  });
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
