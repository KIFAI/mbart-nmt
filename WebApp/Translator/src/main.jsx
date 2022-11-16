import axios from 'axios'
import React from 'react'
import ReactDOM from 'react-dom'
import {App} from './App'
import './index.css'

axios.defaults.baseURL = "http://10.17.23.228:14000";

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
)
