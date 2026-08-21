"""
Script to download raw LOB data from DEIB server

## Args:
* path: destionation path
"""

import argparse
import getpass
import os
import posixpath
import shutil
import zipfile

import paramiko

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SERVER = "131.175.120.196"
USERNAME = "trading"
ROOT = f"/home/{USERNAME}"

# Files/folders on the remote server to copy.
# Use Linux-style paths here, because these are paths on the SSH server.
REMOTE_ITEMS = [
    f"{ROOT}/data_jacopo/cache_FBTP.zip",
    f"{ROOT}/data_jacopo/new_cache_FBTP.zip",
    f"{ROOT}/data_jacopo/cache_FBTS.zip",
    f"{ROOT}/data_jacopo/new_cache_FBTS.zip",
]


# ==============================================================================
# HELPERS
# ==============================================================================


def is_remote_directory(sftp, path):
    """Return True if the remote path is a directory."""
    try:
        return stat_is_directory(sftp.stat(path).st_mode)
    except FileNotFoundError:
        return False


def stat_is_directory(mode):
    """Check whether a Unix file mode represents a directory."""
    import stat

    return stat.S_ISDIR(mode)


def copy_remote_file(sftp, remote_path, local_path):
    """Copy one remote file to the local PC."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"  Copying: {remote_path}")

    sftp.get(remote_path, local_path)


def copy_remote_directory(sftp, remote_path, local_path):
    """Recursively copy a remote directory."""

    os.makedirs(local_path, exist_ok=True)

    for item in sftp.listdir_attr(remote_path):
        remote_item = posixpath.join(remote_path, item.filename)
        local_item = os.path.join(local_path, item.filename)

        if stat_is_directory(item.st_mode):
            copy_remote_directory(sftp, remote_item, local_item)
        else:
            copy_remote_file(sftp, remote_item, local_item)


def copy_remote_item(sftp, remote_path, destination):
    """
    Copy either a remote file or directory.

    The final component of the remote path is preserved.
    """

    name = posixpath.basename(remote_path.rstrip("/"))
    local_path = os.path.join(destination, name)

    try:
        attributes = sftp.stat(remote_path)
    except FileNotFoundError:
        print(f"ERROR: Remote path does not exist: {remote_path}")
        return

    if stat_is_directory(attributes.st_mode):
        print(f"\nCopying folder: {remote_path}")
        copy_remote_directory(sftp, remote_path, local_path)
    else:
        print(f"\nCopying file: {remote_path}")
        copy_remote_file(sftp, remote_path, local_path)


def extract_zip(zip_path, destination):
    """
    Extract a ZIP into destination.

    If the ZIP contains a single top-level folder, that folder is stripped
    so that its contents are extracted directly into destination.
    """

    print("\nUnzipping:")
    print(f"  {zip_path}")
    print(f"  -> {destination}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            members = zip_file.namelist()

            top_level = set()

            for member in members:
                parts = member.replace("\\", "/").split("/")

                if parts and parts[0]:
                    top_level.add(parts[0])

            if len(top_level) == 1:
                top_level_name = next(iter(top_level))

                print(f"  ZIP contains top-level folder: " f"{top_level_name}")

                os.makedirs(destination, exist_ok=True)

                prefix = top_level_name + "/"

                for member in members:
                    normalized = member.replace("\\", "/")

                    # Skip the top-level directory itself.
                    if normalized == top_level_name:
                        continue

                    if not normalized.startswith(prefix):
                        continue

                    relative_path = normalized[len(prefix) :]

                    if not relative_path:
                        continue

                    # There are some garbage nested .zip to filter out
                    if not relative_path.lower().endswith(".parquet"):
                        continue

                    target_path = os.path.join(destination, *relative_path.split("/"))

                    if member.endswith("/"):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)

                        with zip_file.open(member) as source:
                            with open(target_path, "wb") as target:
                                target.write(source.read())

            else:
                print("  ZIP contains multiple top-level items.")

                os.makedirs(destination, exist_ok=True)
                zip_file.extractall(destination)

        print("  Extraction successful.")
        return True

    except zipfile.BadZipFile:
        print("  ERROR: This is not a valid ZIP file.")
        return False

    except Exception as error:
        print(f"  ERROR while extracting: {error}")
        return False


def merge_directories(source, destination):
    """
    Recursively merge source into destination.

    Files from source overwrite files with the same path in destination.
    Directories are merged recursively.
    """

    if not os.path.isdir(source):
        return

    os.makedirs(destination, exist_ok=True)

    for name in os.listdir(source):
        source_path = os.path.join(source, name)
        destination_path = os.path.join(destination, name)

        if os.path.isdir(source_path):
            if os.path.isfile(destination_path):
                os.remove(destination_path)

            merge_directories(source_path, destination_path)

        else:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            shutil.copy2(source_path, destination_path)

            print(f"  Replaced: {destination_path}")


def merge_new_folders(destination):
    """
    Merge every new_* folder into its corresponding folder.

    Example:
        cache_FBTP/     <- existing data
        new_cache_FBTP/ <- updated data

    becomes:
        cache_FBTP/     <- new files overwrite old files

    The new_* folder is deleted after a successful merge.
    """

    print("\nChecking for new_* folders to merge...")

    for name in os.listdir(destination):
        if not name.startswith("new_"):
            continue

        new_folder = os.path.join(destination, name)

        if not os.path.isdir(new_folder):
            continue

        original_name = name[4:]
        original_folder = os.path.join(destination, original_name)

        print("\nMerging:")
        print(f"  {new_folder}")
        print(f"  -> {original_folder}")

        try:
            merge_directories(new_folder, original_folder)

            shutil.rmtree(new_folder)

            print(f"  Deleted: {new_folder}")

        except Exception as error:
            print(f"  ERROR while merging " f"{new_folder}: {error}")


def unzip_files(destination):
    """
    Extract ZIP files.

    ZIPs starting with new_ are extracted into new_* folders.
    They are subsequently merged into the corresponding original
    folder, with new files taking precedence.
    """

    print("\nChecking for ZIP files...")

    zip_files = []

    for root, dirs, files in os.walk(destination):
        for filename in files:
            if filename.lower().endswith(".zip"):
                zip_files.append(os.path.join(root, filename))

    if not zip_files:
        print("No ZIP files found.")
        return

    for zip_path in zip_files:
        zip_filename = os.path.basename(zip_path)
        zip_name = os.path.splitext(zip_filename)[0]

        # Keep the "new_" prefix here.
        # The folders will be merged afterwards.
        final_folder = os.path.join(os.path.dirname(zip_path), zip_name)

        success = extract_zip(zip_path, final_folder)

        if success:
            os.remove(zip_path)
            print(f"  Deleted ZIP: {zip_filename}")

    # Merge new_* folders only after ALL ZIPs have been extracted.
    merge_new_folders(destination)


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Copy data from the remote SSH server.")

    parser.add_argument("destination", help="Local destination directory.")

    args = parser.parse_args()

    destination = os.path.abspath(args.destination.strip('"'))

    os.makedirs(destination, exist_ok=True)

    print("========================================")
    print("       SSH File Transfer")
    print("========================================")
    print()
    print(f"Destination: {destination}")
    print()

    # Password is never stored in the script.
    password = getpass.getpass(f"SSH password for {USERNAME}@{SERVER}: ")

    print()
    print(f"Connecting to {SERVER}...")

    ssh = paramiko.SSHClient()

    # Automatically accept the server's host key if it isn't
    # already known.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname=SERVER, username=USERNAME, password=password)

        print("Connected successfully.")

        sftp = ssh.open_sftp()

        try:
            for remote_item in REMOTE_ITEMS:
                copy_remote_item(sftp, remote_item, destination)

        finally:
            sftp.close()

        unzip_files(destination)

        print()
        print("========================================")
        print("Transfer complete.")
        print("========================================")

    except paramiko.AuthenticationException:
        print("\nERROR: SSH authentication failed.")
        print("Check your username and password.")

    except paramiko.SSHException as error:
        print(f"\nSSH ERROR: {error}")

    except OSError as error:
        print(f"\nSYSTEM ERROR: {error}")

    finally:
        ssh.close()


if __name__ == "__main__":
    main()
