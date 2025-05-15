import { DarkButton } from '../ui/DarkButton'
import { FileUpload, LinkUpload } from '../ui/Inputs';
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

            <div style={{margin: "3vh"}} >
                <DarkButton disabled={!loaded}>
                    Test... Тест!
                </DarkButton>

            </div>
        </div>
    );
}