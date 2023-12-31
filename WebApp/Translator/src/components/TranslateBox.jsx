import axios from "axios";
import TextareaAutosize from "react-textarea-autosize";
import { useEffect, useState } from "react";
import { SourceSelectBox, TargetSelectBox} from "./SelectBox";
import { error, success } from "../utils/notification";
import copy from "copy-to-clipboard";
import { AiFillCopy } from "react-icons/ai";
import { MdClear } from "react-icons/md";
import { Animation } from "./Animation";


export const TranslateBox = () => {

    const [q, setQ] = useState("");
    const [source, setSource] = useState("");
    const [target, setTarget] = useState("");
    const [output, setOutput] = useState("");

    const handleSelectChange = ({ target: { value, id } }) => {
        id === "source" && setSource(value);
        id === "target" && setTarget(value);
    }

    const handleGetRequest = async () => {
        if (q.length < 1) {
            setOutput("");
            return false;
        };
        if (source === "" || target === "") {
            return error("Please select language");
        }
	if (source == target) {
            return error("Please select different language");
        }
        try {
	    const json = JSON.stringify({ q, source, target})
            let res = await axios.post("", { q, source, target, format:"text"})
            res = res.data.translatedText.map(el => el['translated']).join('\n');
            
	    setOutput(res);
        } catch (err) {
            console.log(err);
        }
    }

    const copyToClipboard = (text) => {
        copy(text);
        success("Copied to clipboard!")
    }

    const resetText = () => {
        if (q === "" && output === "") {
            error("Textbox is already empty!")
        } else {
            success("Text removed!")
            setQ("");
            setOutput("");
        }
    }
 
//Debounce Function
    useEffect(() => {

        let timerID = setTimeout(() => {
            handleGetRequest();
        }, 1000);

        return () => {
            clearTimeout(timerID);
        }

    }, [q]);


    return (
        <>
            <div className="mainBox">
                <div>
                    <SourceSelectBox id={'source'} select={handleSelectChange} />
                    <div className="box">
	    		<TextareaAutosize onChange={(e) => { setQ(e.target.value) }} value={q} className="outputResult"></TextareaAutosize>
                    </div>
                    <div className="iconBox">
                        <p>{q.length}/250</p>
                        <AiFillCopy onClick={() => { copyToClipboard(q) }} className="icon" />
                        <MdClear onClick={resetText} className="icon" />
                    </div>
                </div>

                <div>
                    <TargetSelectBox id={'target'} select={handleSelectChange} />
                    <div className="outputResult box">
	    		{output.split('\n').map((item, i) => <p key={i} style={{marginTop:0}}>{item}</p>)}
                    </div>
                    <div className="iconBox">
                        <p>{output.length}/250</p>
                        <AiFillCopy onClick={() => { copyToClipboard(output) }} className="icon" />
                    </div>
                </div>
            </div>

            <Animation />

            <div className="tagLine">
                <p id="madeBykbbank">Made by ❤️ KB Finance AI Center</p>
            </div>
        </>
    );
};
