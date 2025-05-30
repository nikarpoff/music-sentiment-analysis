import { DarkButton } from '../ui/DarkButton'
import { FileUpload, LinkUpload } from '../ui/Inputs';
import { AlertModal, ResultModal } from '../ui/Modal';
import { DarkSelector } from '../ui/DarkSelector'
import { CardCarousel } from '../cards/CardsCarousel';
import classes from "./home.module.css";

import { useState } from "react";

export default function Home() {
    const loadCards = {
        local: "local",
        jamendo: "jamendo",
    }

    const [loaded, setLoaded] = useState('');
    const [model, setModel] = useState('1.0');
    const [currentCard, setCurrentCard] = useState(loadCards.local) 
    const [localFile, setLocalFile] = useState(null);
    const [jamendoLink, setJamendoLink] = useState('');

    const [isAlertOpen, setIsAlertOpen] = useState(false);
    const [alertText, setAlertText] = useState('');

    const [isResultsOpen, setIsResultOpen] = useState(false);
    const [predict, setPredict] = useState('unknown');
    const [probs, setProbs] = useState(null);

    const onCardChange = (card) => {
        switch (card) {
            case loadCards.local:
                if (localFile == null) setLoaded(false);
                else setLoaded(true);
                break;
            case loadCards.jamendo:
                if (jamendoLink.trim()) setLoaded(true);
                else setLoaded(false);
                break;
            default:
                console.log("Invalid card! ", card);
                setLoaded(false);
                return;
        }

        setCurrentCard(card);
    };

    const onModelChange = (select) => {
        setModel(select.target.value)
    }

    const onLocalFileUpload = (file) => {
        setLoaded(true);
        setLocalFile(file);
        console.log('Загружен файл:', localFile);
    };

    const onJamendoLinkChange = (link) => {
        setJamendoLink(link)

        if (link.trim()) {
            setLoaded(true);
        } else {
            setLoaded(false);
        }
    };

    const processPredict = (result) => {
        setPredict(result["predict"]);
        setProbs(result["probs"]);

        setIsResultOpen(true);
    }

    const sendFilePredictionRequest = async (file, model) => {
        const formData = new FormData();
        formData.append("audio", file);
        formData.append("model_version", model);

        try {
            const response = await fetch("http://10.0.0.2/api/predict/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text()
                customAlert(`К сожалению, при отправке запроса произошла ошибка: ${errorText}`)
                console.error(`Ошибка: ${response.status}`);
                return null;
            }

            const result = await response.json();
            processPredict(result);
            return result;
        } catch (error) {
            console.error("Ошибка при отправке запроса:", error);
            customAlert("К сожалению, при отправке запроса произошла ошибка. Проверьте подключение к сети.")
            return null;
        }
    };

    const sendJamendoPredictionRequest = async (link, model) => {
        if (link.indexOf('https://www.jamendo.com/track') === -1) {
            customAlert("Ссылка должна соответствовать формату: 'https://www.jamendo.com/track/<id_track>'")
            return null
        }

        const formData = new FormData();
        formData.append("link", link);
        formData.append("model_version", model);

        try {
            const response = await fetch("http://10.0.0.2/api/predict/link/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text()
                customAlert(`К сожалению, при отправке запроса произошла ошибка: ${errorText}`)
                console.error(`Ошибка: ${response.status}`);
                return null;
            }

            const result = await response.json();
            processPredict(result);
            return result;
        } catch (error) {
            console.error("Ошибка при отправке запроса:", error);
            customAlert("К сожалению, при отправке запроса произошла ошибка. Проверьте подключение к сети.")
            return null;
        }
    };

    const handlePredictClick = () => {
        switch (currentCard) {
            case loadCards.local:
                if (localFile) {
                    sendFilePredictionRequest(localFile, model);
                } else {
                    console.warn("Файл не выбран!");
                    return
                }
                break;
            case loadCards.jamendo:
                if (jamendoLink !== '') {
                    sendJamendoPredictionRequest(jamendoLink, model);
                } else {
                    console.warn("Ссылка не введена!");
                    return
                }
                break;
            default:
                console.error("Неизвестная карточка:", currentCard);
                return;
        }
    };

    const customAlert = (text) => {
        setAlertText(text);
        setIsAlertOpen(true);
    };


    return (
        <div className={classes.home}>
            <p>
                Загрузите аудио-файл любым удобным способом
            </p>

            <CardCarousel onCardChange={onCardChange}>
                <div key={loadCards.local} style={{textAlign: "center", width: "50%"}}>
                    <p>
                        Загрузка из локального хранилища
                    </p>
                    
                    <FileUpload onFileSelect={onLocalFileUpload}/>
                </div>


                <div key={loadCards.jamendo} style={{textAlign: "center", width: "70%"}}>
                    <p>
                        Загрузка по ссылке с 
                        <a style={{marginLeft: "7px"}} href="https://www.jamendo.com/start">
                            Jamendo
                        </a>
                    </p>

                    <LinkUpload onLinkChange={onJamendoLinkChange}/>
                </div>

            </CardCarousel>

            <div style={{display: 'flex', flexDirection: 'row'}} >
                {/* <DarkSelector
                    options={[
                        { value: 'test', label: 'specstr 19B'},
                    ]}
                    value={model}
                    onChange={onModelChange}
                    disabled={!loaded}
                /> */}

                <DarkButton onClick={handlePredictClick} disabled={!loaded}>
                    Отправить
                </DarkButton>
            </div>

            <AlertModal 
                text={alertText} 
                isOpen={isAlertOpen}
                onClose={() => setIsAlertOpen(false)}
            />

            <ResultModal 
                predict={predict}
                probs={probs} 
                isOpen={isResultsOpen}
                onClose={() => setIsResultOpen(false)}
            />
        </div>
    );
}