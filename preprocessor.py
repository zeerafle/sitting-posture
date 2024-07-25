class MoveNetPreprocessor(object):
    """Helper class to preprocess pose sample images for classification."""

    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_path):
        """Creates a preprocessor to detection pose from images and save as CSV.

        Args:
          images_in_folder: Path to the folder with the input images. It should
            follow this structure:
            yoga_poses
            |__ downdog
                |______ 00000128.jpg
                |______ 00000181.bmp
                |______ ...
            |__ goddess
                |______ 00000243.jpg
                |______ 00000306.jpg
                |______ ...
            ...
          images_out_folder: Path to write the images overlay with detected
            landmarks. These images are useful when you need to debug accuracy
            issues.
          csvs_out_path: Path to write the CSV containing the detected landmark
            coordinates and label of each image that can be used to train a pose
            classification model.
        """
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._messages = []

        # Create a temp dir to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()

        # Get list of pose classes and print image statistics
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if
             not n.startswith('.') and os.path.isdir(os.path.join(self._images_in_folder, n))]
        )

    def process(self, per_pose_class_limit=None, detection_threshold=0.1):
        """Preprocesses images in the given folder.
        Args:
          per_pose_class_limit: Number of images to load. As preprocessing usually
            takes time, this parameter can be specified to make the reduce of the
            dataset for testing.
          detection_threshold: Only keep images with all landmark confidence score
            above this threshold.
          detect: A function to detect pose landmarks from an image. The function
            should take an image as input and return the detected pose landmarks.
          draw_prediction_on_image: A function to draw the detected pose landmarks
            on top of the input image. The function should take an image and the
            detected pose landmarks as input and return the image with the landmarks
            drawn on top of it.
        """
        # Loop through the classes and preprocess its images
        for pose_class_name in self._pose_class_names:
            print('Preprocessing', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        pose_class_name + '.csv')
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            # Detect landmarks in each image and write it to a CSV file
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file,
                                            delimiter=',',
                                            quoting=csv.QUOTE_MINIMAL)
                # Get list of images
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                valid_image_count = 0

                # Detect pose landmarks from each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except tf.errors.InvalidArgumentError:
                        self._messages.append('Skipped ' + image_path + '. Invalid image.')
                        continue
                    else:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                        image_height, image_width, channel = image.shape

                    # Skip images that isn't RGB because Movenet requires RGB images
                    if channel != 3:
                        self._messages.append('Skipped ' + image_path +
                                              '. Image isn\'t in RGB format.')
                        continue

                    person = detect(image)

                    # skip keypoints for leg
                    keypoints = person.keypoints[:13]

                    # Save landmarks if all landmarks were detected
                    min_landmark_score = min(
                        [keypoint.score for keypoint in keypoints])
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._messages.append('Skipped ' + image_path +
                                              '. No pose was confidentlly detected.')
                        continue

                    valid_image_count += 1

                    # Draw the prediction result on top of the image for debugging later
                    output_overlay = draw_prediction_on_image(
                        image.numpy().astype(np.uint8), person,
                        close_figure=True, keep_input_size=True)

                    # Write detection result into an image file
                    output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                    # Get landmarks and scale it to the same size as the input image
                    pose_landmarks = np.array(
                        [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                         for keypoint in keypoints],
                        dtype=np.float32)

                    # Write the landmark coordinates to its per-class CSV file
                    coordinates = pose_landmarks.flatten().astype(str).tolist()
                    csv_out_writer.writerow([image_name] + coordinates)

                if not valid_image_count:
                    raise RuntimeError(
                        'No valid images found for the "{}" class.'
                        .format(pose_class_name))

        # Print the error message collected during preprocessing.
        print('\n'.join(self._messages))

        # Combine all per-class CSVs into a single output file
        all_landmarks_df = self._all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)

    def class_names(self):
        """List of classes found in the training dataset."""
        return self._pose_class_names

    def _all_landmarks_as_dataframe(self):
        """Merge all per-class CSVs into a single dataframe."""
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                        class_name + '.csv')
            per_class_df = pl.read_csv(csv_out_path, header=None)

            # Add the labels
            per_class_df['class_no'] = [class_index] * len(per_class_df)
            per_class_df['class_name'] = [class_name] * len(per_class_df)

            # Append the folder name to the filename column (first column)
            per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '')
                                                     + per_class_df[per_class_df.columns[0]].astype(str))

            if total_df is None:
                # For the first class, assign its data to the total dataframe
                total_df = per_class_df
            else:
                # Concatenate each class's data into the total dataframe
                total_df = pl.concat([total_df, per_class_df])

        list_name = [[bodypart.name + '_x', bodypart.name + '_y',
                      bodypart.name + '_score'] for bodypart in BodyPart]
        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ['file_name'] + header_name
        header_map = {total_df.columns[i]: header_name[i]
                      for i in range(len(header_name))}

        total_df.rename(header_map, axis=1, inplace=True)

        return total_df
