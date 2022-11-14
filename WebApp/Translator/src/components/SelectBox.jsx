import './SelectBox.css';

export const SourceSelectBox = ({ id, select }) => {

    return (
        <>
            <div className="select">
                <select id={id} onChange={select}>
                    <option value="">Select Language</option>
		    <option value="en_XX">English</option>
                    <option value="ko_KR">Korean</option>
                </select>
            </div>
        </>
    );
};
export const TargetSelectBox = ({ id, select }) => {

    return (
        <>
            <div className="select">
                <select id={id} onChange={select}>
                    <option value="">Select Language</option>
	    	    <option value="en_XX">English</option>
                    <option value="ko_KR">Korean</option>
                </select>
            </div>
        </>
    );
};
