import { useRef, useState } from 'react';
import classes from './ui.module.css';

export const FileUpload = ({ onFileSelect }) => {
    const fileInputRef = useRef(null);
    const [fileName, setFileName] = useState('');

    const handleClick = () => {
        fileInputRef.current.click();
    };

    const handleChange = (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            setFileName(file.name);
            onFileSelect(file);
        }
    };

    return (
        <div className={classes.uploadContainer} onClick={handleClick}>
            <input
                type="file"
                accept="audio/*"
                ref={fileInputRef}
                onChange={handleChange}
                className={classes.hiddenInput}
            />
            <p>
                {fileName
                    ? `Выбран файл: ${fileName}`
                    : 'Нажмите для загрузки аудиофайла'}
            </p>
        </div>
    );
};

export const LinkUpload = ({ onLinkChange }) => {
    const [link, setLink] = useState('');

    const handleChange = (e) => {
        const newLink = e.target.value
        setLink(newLink);
        onLinkChange(newLink);
    };

    return (
        <input
            type="text"
            value={link}
            onChange={handleChange}
            placeholder="Вставьте ссылку на видео/аудио"
            className={classes.textInput}
        />
    );
};

