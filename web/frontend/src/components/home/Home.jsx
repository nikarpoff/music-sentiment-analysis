import { DarkButton } from '../ui/DarkButton'
import { FileUpload, LinkUpload } from '../ui/Inputs';
import { DarkSelector } from '../ui/DarkSelector'
import { CardCarousel } from '../cards/CardsCarousel';
import classes from "./home.module.css";

import { useState } from "react";

export default function Home() {
    const loadCards = {
        local: "local",
        youtube: "youtube",
        rutube: "rutube",
    }

    const [loaded, setLoaded] = useState('');
    const [model, setModel] = useState('test');
    const [localFile, setLocalFile] = useState(null);
    const [youtubeLink, setYoutubeLink] = useState('');
    const [rutubeLink, setRutubeLink] = useState('');

    const onCardChange = (currentCard) => {
        switch (currentCard) {
            case loadCards.local:
                if (localFile == null) setLoaded(false);
                else setLoaded(true);
                break;
            case loadCards.youtube:
                if (youtubeLink.trim()) setLoaded(true);
                else setLoaded(false);
                break;
            case loadCards.rutube:
                if (rutubeLink.trim()) setLoaded(true);
                else setLoaded(false);
                break;
        }
    };

    const onModelChange = (select) => {
        setModel(select.target.value)
    }

    const onLocalFileUpload = (file) => {
        setLoaded(true);
        setLocalFile(file);
        console.log('Загружен файл:', localFile);
    };

    const onYoutubeLinkChange = (link) => {
        setYoutubeLink(link)

        if (link.trim()) {
            setLoaded(true);
        } else {
            setLoaded(false);
        }
    };

    const onRutubeLinkChange = (link) => {
        setRutubeLink(link)

        if (link.trim()) {
            setLoaded(true);
        } else {
            setLoaded(false);
        }
    };

    const sendPredictionRequest = async (file, model) => {
        const formData = new FormData();
        formData.append("audio", file);
        formData.append("model_version", model);

        try {
            const response = await fetch("http://10.0.0.2/api/predict/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Ошибка: ${response.status}`);
            }

            const result = await response.json();
            console.log(result)
            return result;
        } catch (error) {
            console.error("Ошибка при отправке запроса:", error);
            return null;
        }
    };

    const handlePredictClick = () => {
        if (localFile) {
            sendPredictionRequest(localFile, model);
        } else {
            console.warn("Файл не выбран!");
        }
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


                <div key={loadCards.youtube} style={{textAlign: "center", width: "70%"}}>
                    <p>
                        Загрузка по ссылке с YouTube
                    </p>

                    <LinkUpload onLinkChange={onYoutubeLinkChange}/>
                </div>


                <div key={loadCards.rutube} style={{textAlign: "center", width: "70%"}}>
                    <p>
                        Загрузка по ссылке с RuTube
                    </p>

                    <LinkUpload onLinkChange={onRutubeLinkChange}/>
                </div>
            </CardCarousel>

            {/* <DarkSelector
                options={[
                    { value: 'apple', label: 'Apple' },
                    { value: 'banana', label: 'Banana' },
                ]}
                value={sel}
                onChange={e => setSel(e.target.value)}
            /> */}

            <div style={{display: 'flex', flexDirection: 'row'}} >
                <DarkSelector
                    options={[
                        { value: 'test', label: 'specstr 19B'},
                    ]}
                    value={model}
                    onChange={onModelChange}
                    disabled={!loaded}
                />

                <DarkButton onClick={handlePredictClick} disabled={!loaded}>
                    Сделать предсказание!
                </DarkButton>

            </div>
        </div>
    );
}